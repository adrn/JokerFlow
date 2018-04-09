import numpy as np
import tensorflow as tf
import helpers
from jokerflow import kepler

# These functions are used to transform bounded parameters to parameters to parameters with infinite range
def get_param_for_value(value, min_value, max_value):
    if np.any(value <= min_value) or np.any(value >= max_value):
        raise ValueError("value must be in the range (min_value, max_value)")
    return np.log(value - min_value) - np.log(max_value - value)

def get_value_for_param(param, min_value, max_value):
    return min_value + (max_value - min_value) / (1.0 + np.exp(-param))

def get_bounded_variable(name, value, min_value, max_value, dtype=tf.float64):
    param = tf.Variable(get_param_for_value(value, min_value, max_value), dtype=dtype, name=name + "_param")
    var = min_value + (max_value - min_value) / (1.0 + tf.exp(-param))
    log_jacobian = tf.log(var - min_value) + tf.log(max_value - var) - np.log(max_value - min_value)
    return param, var, tf.reduce_sum(log_jacobian), (min_value, max_value)

# This function constrains a pair of parameters to be a unit vector
def get_unit_vector(name, x_value, y_value, dtype=tf.float64):
    x_param = tf.Variable(x_value, dtype=dtype, name=name + "_x_param")
    y_param = tf.Variable(y_value, dtype=dtype, name=name + "_y_param")
    norm = tf.square(x_param) + tf.square(y_param)
    log_jacobian = -0.5*tf.reduce_sum(norm)
    norm = tf.sqrt(norm)
    x = x_param / norm
    y = y_param / norm
    return x_param, y_param, x, y, log_jacobian

session = tf.InteractiveSession()

with tf.device('/cpu:0'):
    np.random.seed(1234)
    T = tf.float32

    # Simulate some timestamps
    t = np.sort(np.random.uniform(0, 365.0, 1000)).astype(np.float32)
    t_tensor = tf.placeholder(T, name="t")
    feed_dict = {t_tensor: t}

    # We will accumulate the log_prior as we go because we'll need to include the
    # log Jacobians introduced by the reparameterizations
    log_prior = tf.zeros(1, dtype=T)

    # The semi-amplitudes and periods
    n_planets = 2
    log_K = tf.Variable(np.log(np.random.uniform(20, 50, n_planets)), dtype=T)
    true_periods = np.random.uniform(4, 25, n_planets)
    log_P = tf.Variable(np.log(true_periods), dtype=T)

    # Here I'm using a transformation to constrain the omega vector [cos(omega), sin(omega)]
    # to be a unit vector. I got this from the Stan manual. We sample in 'w_x' and 'w_y' and
    # we will get a uniform distribution over omega.
    cw, sw = np.random.randn(2, n_planets)
    w_x, w_y, cosw, sinw, log_jac = get_unit_vector("omega", cw, sw, dtype=T)
    log_prior += log_jac

    # Eccentricity should be constrained to be between 0 and 1.
    e_param, e, log_jac, e_range = get_bounded_variable("e", np.random.uniform(0, 0.2, n_planets), 0.0, 1.0, dtype=T)
    log_prior += log_jac

    # phi is the orbital phase. Like omega above, we'll sample in the unit vector [cos(phi), sin(phi)]
    cp, sp = np.random.randn(2, n_planets)
    phi_x, phi_y, cosphi, sinphi, log_jac = get_unit_vector("phi", cp, sp, dtype=T)
    log_prior += log_jac

    # Parameters for the RV zero-point and jitter
    rv0 = tf.Variable(0.0, dtype=T)
    log_jitter = tf.Variable(np.log(5.0), dtype=T)

    # This is the list of parameters that we will fit for
    var_list = [log_K, log_P, w_x, w_y, e_param, phi_x, phi_y, rv0, log_jitter]

    # Here is an implementation of the RV model
    n = 2*np.pi*tf.exp(-log_P)
    K = tf.exp(log_K)
    w = tf.atan2(sinw, cosw)
    phi = tf.atan2(sinphi, cosphi)
    jitter2 = tf.exp(2*log_jitter)
    t0 = (phi + w) / n

    # Solve Kepler's equation and compute the RV signal for each planet
    M = n * t_tensor[:, None] - (phi + w)
    E = kepler(M, e + tf.zeros_like(M))
    f = 2*tf.atan2(tf.sqrt(1+e)*tf.tan(0.5*E), tf.sqrt(1-e)+tf.zeros_like(E))
    rv_models = rv0 + K * (cosw*(tf.cos(f)+e) - sinw*tf.sin(f))

    # Sum the contributions from each planet
    rv_model = tf.reduce_sum(rv_models, axis=1)

    # Simulate some fake data from the model
    session.run(tf.global_variables_initializer())
    y_true = session.run(rv_model, feed_dict=feed_dict)
    yerr = np.random.uniform(1.0, 5.0, len(t)).astype(np.float32)
    yerr2 = (yerr**2).astype(np.float32)
    y = (y_true + np.sqrt(session.run(jitter2) + yerr2) * np.random.randn(len(t))).astype(np.float32)

    # Compute the likelihood
    log_like = -0.5 * tf.reduce_sum(
        tf.square(y - rv_model) / (yerr2 + jitter2) + tf.log(yerr2 + jitter2)
    )
    log_prob = log_prior + log_like

    grad = tf.gradients(log_prob, var_list)

import time

print(session.run(log_prob, feed_dict=feed_dict))
N  = 500
strt = time.time()
for i in range(N):
    session.run(log_prob, feed_dict=feed_dict)
print((time.time() - strt) / N)

print(session.run(grad, feed_dict=feed_dict))
N = 10
strt = time.time()
for i in range(N):
    session.run(grad, feed_dict=feed_dict)
print((time.time() - strt) / N)

session.close()
