/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  default_random_engine gen;
  num_particles = 100;  // Set the number of particles
  normal_distribution<double> dist_x(x, std[0]); // normal (Gaussian) distribution for x
  normal_distribution<double> dist_y(y, std[1]); // normal (Gaussian) distribution for x
  normal_distribution<double> dist_theta(theta, std[2]); // normal (Gaussian) distribution for theta

  for (int i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = dist_x;
    p.y = dist_y;
    p.theta = dist_theta;
    p.weight = 1;
    particles.push_back(p);
  }

  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  double p_xf, p_yf, p_thetaf;
  for (int i = 0; i < num_particles; ++i) {
    p_x0 = particles[i].x;
    p_y0 = particles[i].y;
    p_theta0 = particles[i].theta;

    if (fabs(yaw_rate) >= 0.00001) {
      p_xf = p_x0 + (velocity / yaw_rate) * (sin(p_theta0 + (yaw_rate * delta_t)) - sin(p_theta0));
      p_yf = p_y0 + (velocity / yaw_rate) * (cos(p_theta0) - cos(p_theta0 + (yaw_rate * delta_t)));
      p_thetaf = p_theta0 + yaw_rate * delta_t;
    }
    else {
      p_xf = p_x0 + velocity * cos(p_theta0) * delta_t;
      p_yf = p_y0 + velocity * sin(p_theta0) * delta_t;
      p_thetaf = p_theta0;
    }

    normal_distribution<double> dist_xf(p_xf, std_pos[0]); // normal (Gaussian) distribution for x
    normal_distribution<double> dist_yf(p_yf, std_pos[1]); // normal (Gaussian) distribution for x
    normal_distribution<double> dist_thetaf(p_thetaf, std_pos[2]); // normal (Gaussian) distribution for theta

    
    particles[i].x = dist_xf(gen);
    particles[i].y = dist_yf(gen);
    particles[i].theta = dist_thetaf(gen);
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++) { // For each observation point
    double pseudo_dist = numeric_limits<double>::max();
    int obs_id = -1;
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;

    for (int j = 0; j < predicted.size(); j++) {
      double pred_x = predicted[j].x;
      double pred_y = predicted[j].y;
      double dist = dist(obs_x, obs_y, pred_x, pred_y); // helper_functions.h
      if (dist < pseudo_dist) {
        pseudo_dist = dist;
        obs_id = predicted[j].id;
      }
    }
    observations[i].id = obs_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    double theta = particles[i].theta;

    // vector to hold map landmark locations within sensor range
    vector<LandmarkObs> predictions;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      if (fabs(landmark_x - particles[i].x) <= sensor_range && fabs(landmark_y - particles[i].y) <= sensor_range) {
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

    // Step 1: Transform from vehicle coordinates to map coordinates
    vector<LandmarkObs> transformed_coords;
    for (int j = 0; j < observations.size(); j++) {
      double xm = cos(theta) * observations[j].x - sin(theta) * observations[j].y + particles[i].x;
      double ym = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + particles[i].y;
      transformed_coords.push_back(LandmarkObs{observations[j].id, xm, ym});
    }

    // Step 2: Associate each transformed observation with landmark identifier L1, L2,..... etc
    dataAssociation(predictions, transformed_coords);

    // Step 3: Calculate the multivariate Gaussian probability density for each observation
    for (int j = 0; j < transformed_coords.size(); j++) {
      double x, y;
      double mu_x = transformed_coords[j].x;
      double mu_y = transformed_coords[j].y;

      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == transformed_coords[j].id) {
          x = predictions[k].x;
          y = predictions[k].y;
        }
      } // end each prediction

      // calculate weight for this observation with multivariate Gaussian
      double sigma_x = std_landmark[0];
      double sigma_y = std_landmark[1];
      double Pxy_term1 = 1 / (2 * M_PI * sigma_x * sigma_y);
      double Pxy_exp1 = pow(x - mu_x, 2) / (2 * pow(sigma_x, 2));
      double Pxy_exp2 = pow(y - mu_y, 2) / (2 * pow(sigma_y, 2));

      double Pxy = Pxy_term1 * exp(-(Pxy_exp1 + Pxy_exp2));

      // product of these is the final probability
      particles[i].weight *= Pxy;
    }   // end each observation
  }     // end each particle
}


void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  vector<double> weights;
  double max_weight = -50;

  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }

  uniform_real_distribution<float> dist_weight(0.0, max_weight);
  uniform_real_distribution<float> dist_index(0.0, num_particles - 1);
  int index = dist_index(gen);
  double beta = 0.0;
  vector<Particle> resampled;

  for (int i = 0; i < num_particles; i++) {
    beta += dist_weight(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled.push_back(particles[index]);
  }

  particles = resampled;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}


string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}