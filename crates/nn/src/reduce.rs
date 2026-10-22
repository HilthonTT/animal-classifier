use std::sync::{Barrier, Mutex, PoisonError};

pub trait Reducer: Sync {
    fn sum(&self, values: &mut [f32]);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoReduce;

impl Reducer for NoReduce {
    fn sum(&self, _values: &mut [f32]) {}
}

#[derive(Debug)]
pub struct SharedReduce {
    barrier: Barrier,
    slots: Vec<Mutex<Vec<f32>>>,
}

impl SharedReduce {
    pub fn new(participants: usize) -> Self {
        Self {
            barrier: Barrier::new(participants),
            slots: (0..participants).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }

    pub fn participant(&self, index: usize) -> Participant<'_> {
        assert!(index < self.slots.len(), "participant {index} out of range");
        Participant {
            shared: self,
            index,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Participant<'a> {
    shared: &'a SharedReduce,
    index: usize,
}

impl Reducer for Participant<'_> {
    fn sum(&self, values: &mut [f32]) {
        let shared = self.shared;
        {
            let mut slot = shared.slots[self.index]
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            slot.clear();
            slot.extend_from_slice(values);
        }
        shared.barrier.wait();
        values.fill(0.0);
        for slot in &shared.slots {
            let slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
            for (v, s) in values.iter_mut().zip(slot.iter()) {
                *v += s;
            }
        }
        shared.barrier.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_participant_sees_the_same_total() {
        let shared = SharedReduce::new(3);
        let results: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..3)
                .map(|i| {
                    let reducer = shared.participant(i);
                    scope.spawn(move || {
                        let mut values = vec![i as f32, 1.0];
                        reducer.sum(&mut values);
                        reducer.sum(&mut values);
                        values
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for values in results {
            assert_eq!(values, vec![9.0, 9.0]);
        }
    }
}
