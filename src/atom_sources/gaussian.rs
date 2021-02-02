//! Atom sources with gaussian velocity distributions.

use super::WeightedProbabilityDistribution;
use crate::atom::*;
use crate::atom_sources::emit::AtomNumberToEmit;
use crate::constant as constant;
use crate::initiate::*;
use nalgebra::Vector3;

extern crate rand;
use rand::distributions::Distribution;
use rand::Rng;

use specs::{
    Component, Entities, Entity, HashMapStorage, Join, LazyUpdate, Read, ReadStorage, System,
    WriteStorage,
};

pub struct GaussianDistributionSourceDefinition {
    pub temperature: Vector3<f64>,
    pub vel_mean: Vector3<f64>,
    pub pos_mean: Vector3<f64>,
    pub pos_std: Vector3<f64>,
    pub mass: f64,
}
impl Component for GaussianDistributionSourceDefinition {
    type Storage = HashMapStorage<Self>;
}

pub struct GaussianDistributionSource {
    vx_distribution: WeightedProbabilityDistribution,
    vy_distribution: WeightedProbabilityDistribution,
    vz_distribution: WeightedProbabilityDistribution,
    x_distribution: WeightedProbabilityDistribution,
    y_distribution: WeightedProbabilityDistribution,
    z_distribution: WeightedProbabilityDistribution,
}
impl Component for GaussianDistributionSource {
    type Storage = HashMapStorage<Self>;
}
impl GaussianDistributionSource {
    fn get_random_velocity<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector3<f64> {
        return Vector3::new(
            self.vx_distribution.sample(rng),
            self.vy_distribution.sample(rng),
            self.vz_distribution.sample(rng),
        );
    }
    fn get_random_position<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector3<f64> {
        return Vector3::new(
            self.x_distribution.sample(rng),
            self.y_distribution.sample(rng),
            self.z_distribution.sample(rng),
        );
    }
}

/// Creates and precalculates a [WeightedProbabilityDistribution](struct.WeightedProbabilityDistribution.html)
/// which can be used to sample values of velocity or position, based on given mean and standrad deviation.
///
/// # Arguments
///
/// `mean`: The mean velocity, in m/s
///
/// `temp`: The temperature of the source in kelvin
fn create_gaussian_distribution(mean: f64, std: f64) -> WeightedProbabilityDistribution {
    // tuple list of (values, weight)
    let mut values = Vec::<f64>::new();
    let mut weights = Vec::<f64>::new();
    
    // precalculate the discretized distribution.
    let n = 1000;
    for i in -n..n {
        let v = (i as f64) / (n as f64) * 5.0 * std;
        let weight = constant::EXP.powf(-(v / std).powf(2.0) / 2.0);
        values.push(v + mean);
        weights.push(weight);
    }

    WeightedProbabilityDistribution::new(values, weights)
}


/// Precalculates the probability distributions for
/// [GaussianDistributionSourceDefinition](struct.GaussianDistributionSourceDefinition.html) and
/// stores the result in a [GaussianDistributionSource](struct.GaussianDistributionSource.html) component.
pub struct PrecalculateForGaussianSourceSystem;
impl<'a> System<'a> for PrecalculateForGaussianSourceSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GaussianDistributionSourceDefinition>,
        WriteStorage<'a, GaussianDistributionSource>,
    );

    fn run(&mut self, (entities, definitions, mut calculated): Self::SystemData) {
        let mut precalculated_data = Vec::<(Entity, GaussianDistributionSource)>::new();
        for (entity, definition, _) in (&entities, &definitions, !&calculated).join() {

            // calculate std of velocites from temperature
            let stdx = (constant::BOLTZCONST * definition.temperature[0]/definition.mass).sqrt();
            let stdy = (constant::BOLTZCONST * definition.temperature[1]/definition.mass).sqrt();
            let stdz = (constant::BOLTZCONST * definition.temperature[2]/definition.mass).sqrt();

            let source = GaussianDistributionSource {
                //velocites
                vx_distribution: create_gaussian_distribution(
                    definition.vel_mean[0],
                    stdx,
                ),
                vy_distribution: create_gaussian_distribution(
                    definition.vel_mean[1],
                    stdy,
                ),
                vz_distribution: create_gaussian_distribution(
                    definition.vel_mean[2],
                    stdz,
                ),
                //positions
                x_distribution: create_gaussian_distribution(
                    definition.pos_mean[0],
                    definition.pos_std[0],
                ),
                y_distribution: create_gaussian_distribution(
                    definition.pos_mean[1],
                    definition.pos_std[1],
                ),
                z_distribution: create_gaussian_distribution(
                    definition.pos_mean[2],
                    definition.pos_std[2],
                ),
            };
            precalculated_data.push((entity, source));
            println!("Precalculated velocity, position and mass distributions for a gaussian source.");
        }

        for (entity, precalculated) in precalculated_data {
            calculated
                .insert(entity, precalculated)
                .expect("Could not add precalculated gaussian source.");
        }
    }
}

pub struct GaussianCreateAtomsSystem;

impl<'a> System<'a> for GaussianCreateAtomsSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, GaussianDistributionSource>,
        ReadStorage<'a, AtomicTransition>,
        ReadStorage<'a, AtomNumberToEmit>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        Read<'a, LazyUpdate>,
    );

    fn run(
        &mut self,
        (entities, sources, atom_infos, numbers_to_emits, positions, masses, updater): Self::SystemData,
    ) {
        let mut rng = rand::thread_rng();
        for (source, atom, number_to_emit, source_position, mass) in (
            &sources,
            &atom_infos,
            &numbers_to_emits,
            &positions,
            &masses,
        )
            .join()
        {
            for _i in 0..number_to_emit.number {
                let new_atom = entities.create();
                let new_vel = source.get_random_velocity(&mut rng);
                let new_pos = source_position.pos + source.get_random_position(&mut rng);
                updater.insert(
                    new_atom,
                    Velocity {
                        vel: new_vel.clone(),
                    },
                );
                updater.insert(
					new_atom,
					Position {
						pos: new_pos.clone(),
					},
				);
                updater.insert(new_atom, Force::new());
                updater.insert(new_atom, mass.clone());
                updater.insert(new_atom, atom.clone());
                updater.insert(new_atom, Atom);
                updater.insert(new_atom, InitialVelocity { vel: new_vel });
                updater.insert(new_atom, NewlyCreated);
            }
        }
    }
}
