extern crate magneto_optical_trap as lib;

use lib::detector;
extern crate specs;

use lib::fileinput::write_file_template;
use lib::simulation_templates::loadfromconfig::create_from_config;
use specs::Builder;

use lib::laser::force::RandomWalkMarker;
use lib::optimization::OptEarly;

use lib::simulation_templates::mot_2d_plus::create;
use lib::atom_sources::oven::VelocityCap;
use lib::destructor::BoundaryMarker;

use specs::RunNow;
use std::time::{Duration, Instant};

//use std::io::stdin;
fn main() {
    //let mut s=String::new();
    //stdin()
    //    .read_line(&mut s)
    //    .expect("Did not enter a correct string");
    let now = Instant::now();
    let (mut world, mut dispatcher) = create_from_config("example.yaml");

    //increase the timestep at the begining of the simulation
    world
        .create_entity()
        .with(OptEarly {
            timethreshold: 2e-4,
            if_opt: false,
        })
        .build();
    //include random walk(Optional)
    //world.create_entity().with(RandomWalkMarker {}).build();

    //include boundary (walls)

    world.create_entity().with(BoundaryMarker {}).build();
    world.create_entity().with(VelocityCap{cap:200.}).build();
    //let (mut world, mut dispatcher) = create();
    for _i in 0..50000 {
        dispatcher.dispatch(&mut world.res);
        world.maintain();
    }
    let mut output = detector::PrintOptResultSystem;
    output.run_now(&world.res);
    println!("time taken to run in ms{}", now.elapsed().as_millis());
    //write_file_template("example.yml")

}
