use crate::rt::{object, Access, Location, Synchronize, VersionVec};
use std::collections::VecDeque;
use std::sync::atomic::Ordering::{Acquire, Release};

#[derive(Debug)]
pub(crate) struct Channel {
    state: object::Ref<State>,
}

#[derive(Debug)]
pub(super) struct State {
    /// Count of messages in the channel.
    msg_cnt: usize,

    /// Live `Sender`s. At 0 the channel is disconnected for the receiver.
    senders: usize,

    /// Whether the `Receiver` is live. Once it drops, a send fails.
    receiver: bool,

    /// Last access that was a send operation.
    last_send_access: Option<Access>,
    /// Last access that was a receive operation.
    last_recv_access: Option<Access>,
    /// Last access whose outcome a send decides: a `try_recv` or the
    /// receiver's drop.
    last_probe_access: Option<Access>,
    /// Last drop of a sender.
    last_hangup_access: Option<Access>,

    /// Every sender drop releases here, as `std`'s `AcqRel` decrement of the
    /// sender count does; a receive that finds the channel disconnected
    /// acquires it.
    senders_gone: Synchronize,
    /// The receiver's drop releases here; a send that fails acquires it.
    receiver_gone: Synchronize,

    /// A synchronization point for synchronizing the sending threads and the
    /// channel.
    ///
    /// The `mpsc` channels have a guarantee that the messages will be received
    /// in the same order in which they were sent. Therefore, if thread `t1`
    /// managed to send `m1` before `t2` sent `m2`, the thread that received
    /// `m2` can be sure that `m1` was already sent and received. In other
    /// words, it is sound for the receiver of `m2` to know that `m1` happened
    /// before `m2`. That is why we have a single `sender_synchronize` for
    /// senders which we use to "timestamp" each message put in the channel.
    /// However, in our example, the receiver of `m1` does not know whether `m2`
    /// was already sent or not and, therefore, by reading from the channel it
    /// should not learn any facts about `happens_before(send(m2), recv(m1))`.
    /// That is why we cannot use single `Synchronize` for the entire channel
    /// and on the receiver side we need to use `Synchronize` per message.
    sender_synchronize: Synchronize,
    /// A synchronization point per message synchronizing the receiving thread
    /// with the channel state at the point when the received message was sent.
    receiver_synchronize: VecDeque<Synchronize>,

    created: Location,
}

/// Actions performed on the Channel.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Action {
    /// Send a message
    MsgSend,
    /// Receive a message
    MsgRecv,
    /// Receive a message if one is queued
    TryRecv,
    /// Drop a sender
    Hangup,
    /// Drop the receiver
    RecvDrop,
}

/// What a `try_recv` found.
pub(crate) enum TryRecv {
    Msg,
    Empty,
    Disconnected,
}

impl Channel {
    pub(crate) fn new(location: Location) -> Self {
        super::execution(|execution| {
            let state = execution.objects.insert_with(
                || State {
                    msg_cnt: 0,
                    senders: 1,
                    receiver: true,
                    last_send_access: None,
                    last_recv_access: None,
                    last_probe_access: None,
                    last_hangup_access: None,
                    senders_gone: Synchronize::new(),
                    receiver_gone: Synchronize::new(),
                    sender_synchronize: Synchronize::new(),
                    receiver_synchronize: VecDeque::new(),
                    created: location,
                },
                |state| {
                    state.msg_cnt = 0;
                    state.senders = 1;
                    state.receiver = true;
                    state.last_send_access = None;
                    state.last_recv_access = None;
                    state.last_probe_access = None;
                    state.last_hangup_access = None;
                    state.senders_gone = Synchronize::new();
                    state.receiver_gone = Synchronize::new();
                    state.sender_synchronize = Synchronize::new();
                    state.receiver_synchronize.clear();
                    state.created = location;
                },
            );

            tracing::trace!(?state, %location, "mpsc::channel");
            Self { state }
        })
    }

    /// Sends a message: false, and nothing queued, once the receiver is gone.
    pub(crate) fn send(&self, location: Location) -> bool {
        self.state.branch_action(Action::MsgSend, location);
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            if !state.receiver {
                state.receiver_gone.sync_load(&mut execution.threads, Acquire);
                return false;
            }
            state.msg_cnt = state.msg_cnt.checked_add(1).expect("overflow");

            state
                .sender_synchronize
                .sync_store(&mut execution.threads, Release);
            state
                .receiver_synchronize
                .push_back(state.sender_synchronize);

            if state.msg_cnt == 1 {
                // Unblock all threads that are blocked waiting on this channel
                let thread_id = execution.threads.active_id();
                for (id, thread) in execution.threads.iter_mut() {
                    if id == thread_id {
                        continue;
                    }

                    let obj = thread
                        .operation
                        .as_ref()
                        .map(|operation| operation.object());

                    if obj == Some(self.state.erase()) {
                        thread.set_runnable();
                    }
                }
            }
            true
        })
    }

    /// Another `Sender`: needs a live one, so it cannot end a disconnect and
    /// takes no step.
    pub(crate) fn clone_sender(&self) {
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.senders = state.senders.checked_add(1).expect("overflow");
        })
    }

    /// A `Sender`'s drop: `std`'s `AcqRel` decrement of the sender count, a
    /// step of its own, since which drop is last decides what it acquires.
    pub(crate) fn drop_sender(&self, location: Location) {
        if super::execution(|execution| execution.threads.is_active()) {
            self.state.branch_action(Action::Hangup, location);
        }
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.senders -= 1;
            if !execution.threads.is_active() {
                return;
            }
            state.senders_gone.sync_load(&mut execution.threads, Acquire);
            state.senders_gone.sync_store(&mut execution.threads, Release);
            if state.senders == 0 {
                // Wake a receiver blocked on the empty channel: it fails now.
                let thread_id = execution.threads.active_id();
                for (id, thread) in execution.threads.iter_mut() {
                    if id != thread_id
                        && thread.operation.as_ref().map(|operation| operation.object())
                            == Some(self.state.erase())
                    {
                        thread.set_runnable();
                    }
                }
            }
        })
    }

    /// The `Receiver`'s drop: queued messages are discarded with it, and every
    /// later send fails.
    pub(crate) fn drop_receiver(&self, location: Location) {
        if super::execution(|execution| execution.threads.is_active()) {
            self.state.branch_action(Action::RecvDrop, location);
        }
        super::execution(|execution| {
            let state = self.state.get_mut(&mut execution.objects);
            state.receiver = false;
            state.msg_cnt = 0;
            state.receiver_synchronize.clear();
            if execution.threads.is_active() {
                state.receiver_gone.sync_store(&mut execution.threads, Release);
            }
        })
    }

    /// Receives a message, blocking while the channel is empty and a sender
    /// is live: false once it is empty with every sender gone.
    pub(crate) fn recv(&self, location: Location) -> bool {
        self.state
            .branch_disable(Action::MsgRecv, self.would_block(), location);
        super::execution(|execution| self.take(execution))
    }

    /// Receives a message if one is queued, without blocking.
    pub(crate) fn try_recv(&self, location: Location) -> TryRecv {
        self.state.branch_action(Action::TryRecv, location);
        super::execution(|execution| {
            if self.get_state(&mut execution.objects).would_block() {
                TryRecv::Empty
            } else if self.take(execution) {
                TryRecv::Msg
            } else {
                TryRecv::Disconnected
            }
        })
    }

    /// Takes the next message: false, acquiring every sender's drop, when the
    /// channel is empty and disconnected.
    fn take(&self, execution: &mut super::Execution) -> bool {
        let state = self.state.get_mut(&mut execution.objects);
        let thread_id = execution.threads.active_id();
        if state.msg_cnt == 0 {
            assert_eq!(state.senders, 0, "expected to be able to read the message");
            state.senders_gone.sync_load(&mut execution.threads, Acquire);
            return false;
        }
        state.msg_cnt -= 1;
        let mut synchronize = state.receiver_synchronize.pop_front().unwrap();
        synchronize.sync_load(&mut execution.threads, Acquire);
        if state.would_block() {
            // Block all **other** threads attempting to read from the channel
            for (id, thread) in execution.threads.iter_mut() {
                if id == thread_id {
                    continue;
                }

                if let Some(operation) = thread.operation.as_ref() {
                    if operation.object() == self.state.erase()
                        && operation.action() == object::Action::Channel(Action::MsgRecv)
                    {
                        let location = operation.location();
                        thread.set_blocked(location, false);
                    }
                }
            }
        }
        true
    }

    fn would_block(&self) -> bool {
        super::execution(|execution| self.get_state(&mut execution.objects).would_block())
    }

    fn get_state<'a>(&self, objects: &'a mut object::Store) -> &'a mut State {
        self.state.get_mut(objects)
    }
}

impl State {
    /// Whether a `recv` must wait: the channel is empty and a sender is live.
    fn would_block(&self) -> bool {
        self.msg_cnt == 0 && self.senders != 0
    }

    pub(super) fn check_for_leaks(&self, index: usize) {
        if self.msg_cnt != 0 {
            if self.created.is_captured() {
                panic!(
                    "Messages leaked.\n  \
                    Channel created: {}\n            \
                    Index: {}\n        \
                    Messages: {}",
                    self.created, index, self.msg_cnt
                );
            } else {
                panic!(
                    "Messages leaked.\n     Index: {}\n  Messages: {}",
                    index, self.msg_cnt
                );
            }
        }
    }

    /// A send is dependent with the sends it is ordered against and with the
    /// probes whose outcome it decides; a `try_recv` with the sends and the
    /// sender drops that decide its outcome; a sender drop with the others,
    /// which decide which one is last, and with the probes.
    pub(super) fn for_each_dependent_access(&self, action: Action, mut f: impl FnMut(&Access)) {
        let (a, b) = match action {
            Action::MsgSend => (&self.last_send_access, &self.last_probe_access),
            Action::MsgRecv => (&self.last_recv_access, &None),
            Action::TryRecv => (&self.last_send_access, &self.last_hangup_access),
            Action::Hangup => (&self.last_hangup_access, &self.last_probe_access),
            Action::RecvDrop => (&self.last_send_access, &None),
        };
        a.iter().chain(b).for_each(&mut f);
    }

    pub(super) fn set_last_access(&mut self, action: Action, path_id: usize, version: &VersionVec) {
        let access = match action {
            Action::MsgSend => &mut self.last_send_access,
            Action::MsgRecv => &mut self.last_recv_access,
            Action::TryRecv | Action::RecvDrop => &mut self.last_probe_access,
            Action::Hangup => &mut self.last_hangup_access,
        };
        Access::set_or_create(access, path_id, version)
    }
}
