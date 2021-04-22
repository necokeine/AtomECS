//! Module for performing linear ramps of quantities.
//!
//! Ramps are characterised by the values the component should take at different keyframes.
//! The component is then linearly interpolated between these values as the simulation proceeds.
//!
//! To ramp a component `T`'s values, add a `Ramp<T>` to the entity. You should also create a
//! `RampUpdateSystem<T>` and add it to the dispatcher.

use specs::{Component, HashMapStorage, Join, ReadExpect, System, WriteStorage};

use crate::integrator::{Step, Timestep};
use std::marker::PhantomData;

pub trait Lerp<T> {
    /// Linearly interpolates from self to b by the given amount (in range 0 to 1).
    fn lerp(&self, b: &T, amount: f64) -> Self;
}

pub struct Ramp<T>
where
    T: Lerp<T> + Component + Clone,
{
    /// Paired list of times and values to have at each time.
    pub keyframes: Vec<(f64, T)>,
    /// prev keyframe in the keyframe list.
    prev: usize,
}

impl<T> Ramp<T>
where
    T: Lerp<T> + Component + Clone,
{
    pub fn get_value(&mut self, current_time: f64) -> T {
        // check if we need to advance cursor
        if !self.at_end() {
            let (t0, _) = &self.keyframes[self.prev + 1];
            if current_time > *t0 {
                self.prev = (self.prev + 1).min(self.keyframes.len() - 1);
            }
        }
        // if at end, return last frame value.
        if self.at_end() {
            let (_, last) = &self.keyframes[self.prev];
            return last.clone();
        }

        // not on last element, lerp between
        let (t1, val_a) = &self.keyframes[self.prev];
        let (t2, val_b) = &self.keyframes[self.prev + 1];
        let amount = (current_time - t1) / (t2 - t1);
        return val_a.lerp(&val_b, amount);
    }

    fn at_end(&self) -> bool {
        return self.prev == self.keyframes.len() - 1;
    }
}

impl<T> Component for Ramp<T>
where
    T: Lerp<T> + Component + Sync + Send + Clone,
{
    type Storage = HashMapStorage<Self>;
}

pub struct RampUpdateSystem<T>
where
    T: Component,
    T: Lerp<T>,
{
    ramped: PhantomData<T>,
}

impl<'a, T> System<'a> for RampUpdateSystem<T>
where
    T: Lerp<T> + Component + Sync + Send + Clone,
{
    type SystemData = (
        WriteStorage<'a, T>,
        WriteStorage<'a, Ramp<T>>,
        ReadExpect<'a, Timestep>,
        ReadExpect<'a, Step>,
    );

    fn run(&mut self, (mut comps, mut ramps, timestep, step): Self::SystemData) {
        let current_time = step.n as f64 * timestep.delta;

        for (ramp, comp) in (&mut ramps, &mut comps).join() {
            comp.clone_from(&ramp.get_value(current_time));
        }
    }
}

pub mod tests {
    use super::*;
    extern crate specs;
    use specs::{Component, HashMapStorage};

    #[derive(Clone)]
    struct ALerpComp {
        value: f64,
    }

    impl Component for ALerpComp {
        type Storage = HashMapStorage<Self>;
    }

    impl Lerp<ALerpComp> for ALerpComp {
        fn lerp(&self, other: &Self, amount: f64) -> Self {
            ALerpComp {
                value: self.value * (1.0 - amount) + other.value * amount,
            }
        }
    }

    #[test]
    fn test_ramp() {
        let comp_a = ALerpComp { value: 0.0 };
        let comp_b = ALerpComp { value: 1.0 };

        let mut frames = Vec::new();
        frames.push((0.0, comp_a.clone()));
        frames.push((1.0, comp_b.clone()));
        let mut ramp = Ramp {
            prev: 0,
            keyframes: frames,
        };

        let a = ramp.get_value(0.0);
        assert!(
            a.value < std::f64::EPSILON,
            "incorrect: a.value={}, target={}",
            a.value,
            0.0
        );
        let b = ramp.get_value(0.5);
        assert!(
            (0.5 - b.value).abs() < std::f64::EPSILON,
            "incorrect: value={}, target={}",
            b.value,
            0.5
        );
        let c = ramp.get_value(1.0);
        assert!(
            (1.0 - c.value).abs() < std::f64::EPSILON,
            "incorrect: value={}, target={}",
            c.value,
            1.0
        );
        let d = ramp.get_value(2.0);
        assert!(
            (1.0 - d.value).abs() < std::f64::EPSILON,
            "incorrect: value={}, target={}",
            d.value,
            1.0
        );
    }
}
