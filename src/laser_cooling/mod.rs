//! A module that implements systems and components for calculating optical scattering forces in AtomECS.

use crate::atom::AtomicTransition;
use crate::constant;
use crate::initiate::NewlyCreated;
use crate::integrator::INTEGRATE_POSITION_SYSTEM_NAME;
use crate::laser::index::LaserIndex;
use crate::laser::BEAM_LIMIT;
use crate::ramp::Lerp;
use serde::{Deserialize, Serialize};
use specs::prelude::*;

pub mod doppler;
pub mod force;
pub mod photons_scattered;
pub mod rate;
pub mod repump;
pub mod sampler;
pub mod twolevel;

/// A component representing light properties used for laser cooling.
///
/// Currently only holds the information about polarization and wavelength
/// and works as a marker for all laser cooling processes. This will be
/// split into different components in a future version.
#[derive(Deserialize, Serialize, Clone, Copy)]
pub struct CoolingLight {
    /// Polarisation of the laser light, 1 for +, -1 for -,
    ///
    /// Note that the polarization is defined by the quantization vector (e.g. magnetic field)
    /// and not (always) in direction of the wavevector. Look at the given examples of 3D-MOT
    /// simulations to see a working example if unsure.
    ///
    /// Currently this is an integer value since every partial polarization can be expressed
    /// as a superposition of fully polarized beams. It  is possible that this will be
    /// changed to a non-integer value in the future.
    pub polarization: i32,

    /// wavelength of the laser light, in SI units of m.
    pub wavelength: f64,
}

impl Lerp<CoolingLight> for CoolingLight {
    fn lerp(&self, b: &CoolingLight, amount: f64) -> Self {
        CoolingLight {
            polarization: self.polarization,
            wavelength: self.wavelength - (self.wavelength - b.wavelength) * amount,
        }
    }
}

impl CoolingLight {
    /// Frequency of the cooling light in units of Hz
    pub fn frequency(&self) -> f64 {
        constant::C / self.wavelength
    }

    /// Wavenumber of the cooling light, in units of 2pi inverse metres.
    pub fn wavenumber(&self) -> f64 {
        2.0 * constant::PI / self.wavelength
    }

    /// Creates a `CoolingLight` component from the desired atomic species.
    ///
    /// # Arguments
    ///
    /// `species`: The atomic species to take the base wavelength from.
    ///
    /// `detuning`: Detuning of the laser from transition in units of MHz
    ///
    /// `polarization`: Polarization of the cooling beam.
    pub fn for_species(species: AtomicTransition, detuning: f64, polarization: i32) -> Self {
        let freq = species.frequency + detuning * 1.0e6;
        CoolingLight {
            wavelength: constant::C / freq,
            polarization,
        }
    }
}
impl Component for CoolingLight {
    type Storage = HashMapStorage<Self>;
}

/// A system which attaches components required for optical scattering force calculation to newly created atoms.
///
/// They are recognized as newly created if they are associated with
/// the `NewlyCreated` component.
pub struct AttachLaserCoolingComponentsToNewlyCreatedAtomsSystem;

impl<'a> System<'a> for AttachLaserCoolingComponentsToNewlyCreatedAtomsSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, NewlyCreated>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (ent, newly_created, updater): Self::SystemData) {
        for (ent, _) in (&ent, &newly_created).join() {
            updater.insert(
                ent,
                doppler::DopplerShiftSamplers {
                    contents: [doppler::DopplerShiftSampler::default(); BEAM_LIMIT],
                },
            );
            updater.insert(
                ent,
                sampler::LaserDetuningSamplers {
                    contents: [sampler::LaserDetuningSampler::default(); BEAM_LIMIT],
                },
            );
            updater.insert(
                ent,
                rate::RateCoefficients {
                    contents: [rate::RateCoefficient::default(); BEAM_LIMIT],
                },
            );
            updater.insert(ent, twolevel::TwoLevelPopulation::default());
            updater.insert(ent, photons_scattered::TotalPhotonsScattered::default());
            updater.insert(
                ent,
                photons_scattered::ExpectedPhotonsScatteredVector {
                    contents: [photons_scattered::ExpectedPhotonsScattered::default(); BEAM_LIMIT],
                },
            );
            updater.insert(
                ent,
                photons_scattered::ActualPhotonsScatteredVector {
                    contents: [photons_scattered::ActualPhotonsScattered::default(); BEAM_LIMIT],
                },
            );
        }
    }
}

/// A system that attaches `LaserIndex` components to entities which have `CoolingLight` but no index.
pub struct AttachIndexToCoolingLightSystem;
impl<'a> System<'a> for AttachIndexToCoolingLightSystem {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, CoolingLight>,
        ReadStorage<'a, LaserIndex>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (ent, cooling_light, cooling_light_index, updater): Self::SystemData) {
        for (ent, _, _) in (&ent, &cooling_light, !&cooling_light_index).join() {
            updater.insert(ent, LaserIndex::default());
        }
    }
}

/// Adds the systems required by the module to the dispatcher.
///
/// #Arguments
///
/// `builder`: the dispatch builder to modify
///
/// `deps`: any dependencies that must be completed before the systems run.
pub fn add_systems_to_dispatch(builder: &mut DispatcherBuilder<'static, 'static>, deps: &[&str]) {
    builder.add(
        AttachLaserCoolingComponentsToNewlyCreatedAtomsSystem,
        "attach_laser_cooling_components",
        deps,
    );
    builder.add(
        photons_scattered::InitialiseExpectedPhotonsScatteredVectorSystem,
        "initialise_expected_photons",
        deps,
    );
    builder.add(
        rate::InitialiseRateCoefficientsSystem,
        "initialise_rate_coefficients",
        deps,
    );
    builder.add(
        doppler::CalculateDopplerShiftSystem,
        "calculate_doppler_shift",
        &["index_lasers"],
    );
    builder.add(
        sampler::CalculateLaserDetuningSystem,
        "calculate_laser_detuning",
        &["calculate_doppler_shift", "zeeman_shift", "index_lasers"],
    );
    builder.add(
        rate::CalculateRateCoefficientsSystem,
        "calculate_rate_coefficients",
        &["calculate_laser_detuning", "initialise_rate_coefficients"],
    );
    builder.add(
        twolevel::CalculateTwoLevelPopulationSystem,
        "calculate_twolevel",
        &["calculate_rate_coefficients", "fill_laser_sampler_masks"],
    );
    builder.add(
        photons_scattered::CalculateMeanTotalPhotonsScatteredSystem,
        "calculate_total_photons",
        &["calculate_twolevel"],
    );
    builder.add(
        photons_scattered::CalculateExpectedPhotonsScatteredSystem,
        "calculate_expected_photons",
        &[
            "calculate_total_photons",
            "fill_laser_sampler_masks",
            "initialise_expected_photons",
        ],
    );
    builder.add(
        photons_scattered::CalculateActualPhotonsScatteredSystem,
        "calculate_actual_photons",
        &["calculate_expected_photons"],
    );
    builder.add(
        force::CalculateAbsorptionForcesSystem,
        "calculate_absorption_forces",
        &["calculate_actual_photons", INTEGRATE_POSITION_SYSTEM_NAME],
    );
    builder.add(
        repump::RepumpSystem,
        "repump",
        &["calculate_absorption_forces"],
    );
    builder.add(
        force::ApplyEmissionForceSystem,
        "calculate_emission_forces",
        &[
            "calculate_absorption_forces",
            INTEGRATE_POSITION_SYSTEM_NAME,
        ],
    );
    builder.add(
        AttachIndexToCoolingLightSystem,
        "attach_cooling_index",
        deps,
    );
}

#[cfg(test)]
pub mod tests {

    use super::*;
    use assert_approx_eq::assert_approx_eq;
    #[test]
    fn test_add_index_component_to_cooling_lights() {
        let mut test_world = World::new();
        test_world.register::<LaserIndex>();
        test_world.register::<CoolingLight>();

        let test_entity = test_world
            .create_entity()
            .with(CoolingLight {
                polarization: 1,
                wavelength: 780e-9,
            })
            .build();

        let mut system = AttachIndexToCoolingLightSystem;
        system.run_now(&test_world);
        test_world.maintain();

        assert!(
            !test_world
                .read_storage::<LaserIndex>()
                .get(test_entity)
                .is_none()
        );
    }

    #[test]
    fn test_for_species() {
        let detuning = 12.0;
        let light = CoolingLight::for_species(AtomicTransition::rubidium(), detuning, 1);
        assert_approx_eq!(
            light.frequency(),
            AtomicTransition::rubidium().frequency + 1.0e6 * detuning
        );
    }
}
