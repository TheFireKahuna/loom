//! A stub for `std::sync::mpsc`.

use crate::rt;

/// Mock implementation of `std::sync::mpsc::channel`.
#[track_caller]
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let location = location!();
    let (sender_channel, receiver_channel) = std::sync::mpsc::channel();
    let channel = std::sync::Arc::new(rt::Channel::new(location));
    let sender = Sender {
        object: std::sync::Arc::clone(&channel),
        sender: sender_channel,
    };
    let receiver = Receiver {
        object: std::sync::Arc::clone(&channel),
        receiver: receiver_channel,
    };
    (sender, receiver)
}

#[derive(Debug)]
/// Mock implementation of `std::sync::mpsc::Sender`.
pub struct Sender<T> {
    object: std::sync::Arc<rt::Channel>,
    sender: std::sync::mpsc::Sender<T>,
}

impl<T> Sender<T> {
    /// Attempts to send a value on this channel, returning it back if it could
    /// not be sent.
    #[track_caller]
    pub fn send(&self, msg: T) -> Result<(), std::sync::mpsc::SendError<T>> {
        let delivered = self.object.send(location!());
        let result = self.sender.send(msg);
        assert_eq!(delivered, result.is_ok(), "[loom internal bug] send disagrees with std");
        result
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Sender<T> {
        self.object.clone_sender();
        Sender {
            object: std::sync::Arc::clone(&self.object),
            sender: self.sender.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    #[track_caller]
    fn drop(&mut self) {
        self.object.drop_sender(location!());
    }
}

#[derive(Debug)]
/// Mock implementation of `std::sync::mpsc::Receiver`.
pub struct Receiver<T> {
    object: std::sync::Arc<rt::Channel>,
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> Receiver<T> {
    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up.
    #[track_caller]
    pub fn recv(&self) -> Result<T, std::sync::mpsc::RecvError> {
        if !self.object.recv(location!()) {
            return Err(std::sync::mpsc::RecvError);
        }
        Ok(self.received())
    }
    /// Attempts to wait for a value on this receiver, returning an error if the
    /// corresponding channel has hung up, or if it waits more than `timeout`.
    pub fn recv_timeout(
        &self,
        _timeout: std::time::Duration,
    ) -> Result<T, std::sync::mpsc::RecvTimeoutError> {
        unimplemented!("std::sync::mpsc::Receiver::recv_timeout is not supported yet in Loom.")
    }

    /// Attempts to return a pending value on this receiver without blocking.
    #[track_caller]
    pub fn try_recv(&self) -> Result<T, std::sync::mpsc::TryRecvError> {
        match self.object.try_recv(location!()) {
            rt::TryRecv::Msg => Ok(self.received()),
            rt::TryRecv::Empty => Err(std::sync::mpsc::TryRecvError::Empty),
            rt::TryRecv::Disconnected => Err(std::sync::mpsc::TryRecvError::Disconnected),
        }
    }

    /// The message the model just received, which `std`'s channel holds too.
    fn received(&self) -> T {
        self.receiver
            .try_recv()
            .expect("[loom internal bug] the model received a message std does not hold")
    }
}

impl<T> Drop for Receiver<T> {
    #[track_caller]
    fn drop(&mut self) {
        self.object.drop_receiver(location!());
    }
}
