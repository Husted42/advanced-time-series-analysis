
# After importing the dataset we rename it to df
df <- ex1_rainfallrunoff

# Converting timestamp to datetime
df$timestamp <- as.POSIXct(
  df$timestamp,
  format = "%Y-%m-%d %H:%M:%S",
  tz = "UTC"
)


# Making a new column containing the time in hours since the first observation
df$time_hours <- as.numeric(
  difftime(df$timestamp, df$timestamp[1], units = "hours")
)

# Inspecting the data
head(df)
str(df)
summary(df)
colSums(is.na(df))  # no missing values


##### Plotting rainfall and stormwater #####

#install.packages("ggplot2")

library(ggplot2)

# Plotting rainfall
ggplot(df, aes(x = time_hours, y = rainfall)) +
  geom_line(color = "blue") +
  labs(
    title = "Rainfall input",
    x = "Hours since first observation (08:00)",
    y = "Rainfall"
  ) +
  scale_x_continuous(breaks = 0:10) +
  theme_minimal()

# Plotting stormwater
ggplot(df, aes(x = time_hours, y = stormwater)) +
  geom_line(color = "blue") +
  labs(
    title = "Observed stormwater",
    x = "Hours since first observation (08:00)",
    y = "Stormwater"
  ) +
  scale_x_continuous(breaks = 0:10) +
  theme_minimal()


#######################################################
# Question 1.1
#######################################################

#install.packages("ctsmTMB")
library(ctsmTMB)


# The data we will be using
data <- df[, c("time_hours", "rainfall", "stormwater")]
colnames(data)[1] <- "t"  # Renaming the time_hours to "t"
head(data)


# Creating the model
model <- ctsmTMB$new()

# Defining the reservoir equations
model$addSystem(
  dx1 ~ A*rainfall*dt - x1/K*dt + sigma*dw1,
  dx2 ~ (1/K)*x1 * dt + sigma * dw2
)

# The observed stormwater corresponds to reservoir 2
model$addObs(stormwater ~ x2)

# Observation noise variance
model$setVariance(stormwater ~ sigma_y^2)

# Defining rainfall as the input
model$addInput(rainfall)

# Looking at the defined model
print(model)


##### Defining initial parameter values #####

# Calculate time intervals in hours
time_step_hours <- diff(data$t)

# Approximate total rainfall input
total_rainfall <- sum(
  head(data$rainfall, -1) * time_step_hours
)

# Observed change in stormwater
delta_stormwater <- tail(data$stormwater, 1) - data$stormwater[1]

# Initial guesses
delta_stormwater / total_rainfall # 5.161454
A_start <- 5
K_start <- 2
sigma_start <- 1
sigma_y_start <- sd(diff(tail(data$stormwater, 60))) / sqrt(2)

# Setting parameters and bounds
model$setParameter(
  A = c(
    initial = A_start,
    lower = 0.001,
    upper = 100
  ),
  K = c(
    initial = K_start,
    lower = 0.001,
    upper = 50
  ),
  sigma = c(
    initial = sigma_start,
    lower = 0.0001,
    upper = 30
  ),
  sigma_y = c(
    initial = sigma_y_start,
    lower = 0.0001,
    upper = 10
  )
)

# Set initial state means and covariance
model$setInitialState(
  list(
    x0 = c(x1 = 0, x2 = data$stormwater[1]),
    p0 = diag(0.01, 2)
  )
)

# Check parameter specifications
print(model$getParameters())

##### Maximum likelihood estimation #####

fit <- model$estimate(
  data = data,
  method = "lkf", # using the linear kalman filter since the two-state model is linear in the state variables
  ode.solver = "rk4",
  ode.timestep = 1/60
)

# Print estimation results
print(fit)

# Estimated parameters
print(fit$par.fixed)

# Log-likelihood
cat("Log-likelihood:", -fit$nll, "\n") # -217.0542

# Standard errors
print(fit$sd.fixed)

# Check convergence
print(fit$convergence) # 0, So the estimation converged successfully



###############################################
# Question 1.2
###############################################

# Defining a function to estimate a model for a given n
fit_reservoir <- function(n) {
  
  N <- n + 1  # Number of states
  
  m <- ctsmTMB$new() # Creating the model
  
  # Generating reservoir equations
  equations <- lapply(1:N, function(i) {
    if (i == 1) {
      eq <- sprintf(
        "dx1 ~ A*rainfall*dt - %d/K*x1*dt + sigma*dw1",
        n
      )
      
    } else if (i == N) {
      eq <- sprintf(
        "dx%d ~ %d/K*x%d*dt + sigma*dw%d",
        i, n, i-1, i
      )
      
    } else {
      eq <- sprintf(
        "dx%d ~ %d/K*x%d*dt - %d/K*x%d*dt + sigma*dw%d",
        i, n, i-1, n, i, i
      )
    }
    
    as.formula(eq) # converts the character string into an R formula
  })
  
  # Add equations to model
  do.call(m$addSystem, equations)
  
  # Observe the last reservoir
  m$addObs(
    as.formula(paste0("stormwater ~ x", N))
  )
  
  m$setVariance(stormwater ~ sigma_y^2)
  m$addInput(rainfall)
  
  # Initial parameter values from Q1.1
  p <- fit$par.fixed
  
  m$setParameter(
    A = c(initial = unname(p["A"]),
          lower = 0.001, upper = 100),
    
    K = c(initial = unname(p["K"]),
          lower = 0.001, upper = 50),
    
    sigma = c(initial = unname(p["sigma"]),
              lower = 0.0001, upper = 30),
    
    sigma_y = c(initial = unname(p["sigma_y"]),
                lower = 0.0001, upper = 10)
  )
  
  # Initial state values
  x0 <- rep(0, N)
  x0[N] <- data$stormwater[1]
  names(x0) <- paste0("x", 1:N)
  
  m$setInitialState(
    list(x0 = x0, p0 = diag(0.01, N))
  )
  
  # Estimate model
  estimation <- m$estimate(
    data = data,
    method = "lkf",
    ode.solver = "rk4",
    ode.timestep = 1/60
  )
  
  return(list(model = m, fit = estimation))
}

##### Estimating models for n= 1,...,7 #####

n_values <- 1:7

# Initialize list to store all the fitted models
models <- list()

# Start results table
results <- data.frame(
  n = n_values,
  States = n_values + 1,
  NLL = NA_real_,
  Convergence = NA_integer_
)

for (i in seq_along(n_values)) {
  
  n <- n_values[i]
  
  cat("\nEstimating model with n =", n, "\n")
  
  if (n == 1) {
    # We reuse the existing two-state model
    models[[as.character(n)]] <- list(
      model = model,
      fit = fit
    )
  } else {
    # Fitting a new model with n+1 states
    models[[as.character(n)]] <- tryCatch(
      fit_reservoir(n),
      error = function(e) {
        message("Estimation failed: ", conditionMessage(e))
        NULL
      }
    )
  }
  
  result <- models[[as.character(n)]]
  
  if (!is.null(result)) {
    results$NLL[i] <- result$fit$nll
    results$Convergence[i] <- result$fit$convergence
  }
}

print(results)
#   n States      NLL Convergence
# 1 1      2 217.0542           0
# 2 2      3 167.0822           0
# 3 3      4 150.9886           0
# 4 4      5 161.2484           0
# 5 5      6 168.7390           0
# 6 6      7 173.4299           0
# 7 7      8 176.5572           0


##### Plotting the negative log-likelihood as a function of the n-values #####

ggplot(results, aes(x = States, y = NLL)) +
  geom_line(color = "blue") +
  geom_point(color = "blue", size = 3) +
  scale_x_continuous(breaks = n_values) +
  labs(
    x = "States (N)",
    y = "Negative log-likelihood"
  ) +
  theme_minimal()


#################################################
# Question 1.3
#################################################

# The best model (n = 3, N = 4 states)
best_fit <- models[["3"]]$fit
best_model <- models[["3"]]$model

# N=4 (n=3) is the best number of states. 
# Printing the parameter estimates from this model
print(best_fit$par.fixed) # 5.0451270 2.0460209 0.2790511 0.2907803 

# The standard errors
print(best_fit$sd.fixed) # 0.110706281 0.036782200 0.049225592 0.009041153 

# Checking the the model converged
print(best_fit$convergence) # 0. So it converged



#######################################################
# Question 1.4
#######################################################

# Extracting the covariance matrix of parameter estimates
cov_matrix <- best_fit$cov.fixed

# Converting the covariance matrix to correlation matrix
cor_matrix <- cov2cor(cov_matrix)

# Printing correlation matrix rounded to 3 decimals
print(round(cor_matrix, 3))
#             A      K
#   A       1.000  0.414
#   K       0.414  1.000


##########################################################
# Question 1.5
##########################################################

# Plotting residual analysis plots using the built-in plotting function from ctsmTMB
plot(best_fit)


##### Plotting the simulated values #####

# Defining the simulation data as the data with the observations removed
sim_data <- data
sim_data$stormwater <- NA_real_

# Simulating 100 stochastic trajectories
sim <- best_model$simulate(
  data = sim_data,
  pars = best_fit$par.fixed,
  method = "ekf", # Now we use the extendes kalmen filter, because it was not working with the linear
  ode.solver = "rk4",
  ode.timestep = 1/60,
  simulation.timestep = 1/60,
  n.sims = 100,
  cpp.seeds = c(123, 456)
)

# Extracting simulated stormwater observations
sim_stormwater <- sim$observations$stormwater$i0

# Extracting simulation times
sim_time <- sim$times$i0[, "t.j"]

# Calculat theing mean of simulated trajectories
sim_mean <- rowMeans(sim_stormwater)

# Calculating the 95% confidence interval
sim_interval <- t(apply(
  sim_stormwater,
  1,
  quantile,
  probs = c(0.025, 0.975)
))

# Creating a dataframe for plotting
sim_plot <- data.frame(
  t = sim_time,
  observed = data$stormwater,
  simulated = sim_mean,
  lower = sim_interval[, 1],
  upper = sim_interval[, 2]
)


##### Plotting the simulated and observed values #####

ggplot(sim_plot, aes(x = t)) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper),
    fill = "lightblue",
    alpha = 0.4
  ) +
  geom_line(
    aes(y = observed, color = "Observed"),
    linewidth = 0.8
  ) +
  geom_line(
    aes(y = simulated, color = "Simulated mean"),
    linewidth = 0.8
  ) +
  scale_color_manual(
    values = c(
      "Observed" = "red",
      "Simulated mean" = "blue"
    )
  ) +
  scale_x_continuous(
    breaks = 0:10,
    labels = sprintf("%02d:00", 8:18) # making the x-label as clock times after 08:00
  ) +
  labs(
    x = "Time",
    y = "Runoff",
    color = ""
  ) +
  theme_minimal()
