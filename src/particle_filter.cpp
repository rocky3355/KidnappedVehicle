#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES 100


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < NUM_PARTICLES; ++i) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(random);
		particle.y = dist_y(random);
		particle.theta = dist_theta(random);
		particle.weight = 1.0;
		
		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double dt, double std_pos[], double velocity, double yaw_rate) {
	double temp = velocity / yaw_rate;
	double delta_theta = yaw_rate * dt;

	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(delta_theta, std_pos[2]);

	for (Particle& particle : particles) {
		double delta_x = temp * (std::sin(particle.theta + delta_theta) - std::sin(particle.theta));
		double delta_y = temp * (std::cos(particle.theta) - std::cos(particle.theta + delta_theta));

		particle.x += delta_x + dist_x(random);
		particle.y += delta_y + dist_y(random);
		particle.theta += dist_theta(random);
	}
}

void ParticleFilter::transform(const std::vector<LandmarkObs>& observations, const Particle& p, std::vector<LandmarkObs>& observations_map) {
	for (const LandmarkObs& obs : observations) {
		LandmarkObs obs_map;
		double sin_theta = std::sin(p.theta);
		double cos_theta = std::cos(p.theta);
		obs_map.x = cos_theta * obs.x - sin_theta * obs.y + p.x;
		obs_map.y = sin_theta * obs.x + cos_theta * obs.y + p.y;
		observations_map.push_back(obs_map);
	}
}

void ParticleFilter::filterLandmarks(const Map &map_landmarks, const Particle& p, double sensor_range_squared, std::vector<Map::single_landmark_s>& valid_landmarks) {
	for (const Map::single_landmark_s& lm : map_landmarks.landmark_list) {
		double distance = calcDistanceSquared(p.x, p.y, lm);
		if (distance <= sensor_range_squared) {
			valid_landmarks.push_back(lm);
		}
	}
}

void ParticleFilter::dataAssociation(const std::vector<Map::single_landmark_s>& landmarks, std::vector<LandmarkObs>& observations) {
	for (LandmarkObs& obs : observations) {
		int id = -1;
		double min_distance = std::numeric_limits<double>::max();
		for (uint i = 0; i < landmarks.size(); i++) {
			double distance = calcDistanceSquared(obs, landmarks[i]);
			if (distance < min_distance) {
				id = i;
				min_distance = distance;
			}
		}
		obs.id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	double sensor_range_squared = sensor_range * sensor_range;

	for (Particle& particle : particles) {
		// Transform from vehicle to map coordinate system
		std::vector<LandmarkObs> observations_map;
		transform(observations, particle, observations_map);

		std::vector<Map::single_landmark_s> valid_landmarks;
		filterLandmarks(map_landmarks, particle, sensor_range_squared, valid_landmarks);

		// Associate observations to nearest landmarks
		dataAssociation(valid_landmarks, observations_map);

		// Calculate new particle weights
		double total_prob = 1.0;
		double dev_x = std_landmark[0];
		double dev_y = std_landmark[1];
		for (const LandmarkObs& obs : observations_map) {
			const Map::single_landmark_s& lm = valid_landmarks[obs.id];
			double temp1 = (obs.x - lm.x_f) * (obs.x - lm.x_f) / (2.0 * dev_x * dev_x);
			double temp2 = (obs.y - lm.y_f) * (obs.y - lm.y_f) / (2.0 * dev_y * dev_y);
			double p = 1.0 / (2.0 * M_PI * dev_x * dev_y) * exp(-temp1 - temp2);
			total_prob *= p;
		}

		particle.weight = total_prob;
	}
}

void ParticleFilter::resample() {
	// Caluclate sum of weights
	double total_weight = 0.0;
	for (const Particle& p : particles) {
		total_weight += p.weight;
	}

	// Normalize weights
	std::vector<double> alpha;
	for (Particle& p : particles) {
		p.weight /= total_weight;
		alpha.push_back(p.weight);
	}

	// Resample with respect to alpha
	std::vector<Particle> resampled_particles;
	std::discrete_distribution<> dist(alpha.begin(), alpha.end());
	for (uint i = 0; i < particles.size(); i++) {
		resampled_particles.push_back(particles[dist(random)]);
	}

	particles = resampled_particles;
}

void ParticleFilter::setAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
