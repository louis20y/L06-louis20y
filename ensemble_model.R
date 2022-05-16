# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("model_info/knn_res.rda")
load("model_info/svm_res.rda")
load("model_info/lin_reg_res.rda")


# Load split data object & get testing data
load("data/wildfires_split.rda")

wildfires_test <- wildfires_split %>% testing()

save(wildfires_test, file = "model_info/wildfires_test.rda")

# Create data stack ----

stack <- stacks() %>% 
  add_candidates(lin_reg_res) %>% 
  add_candidates(knn_res) %>% 
  add_candidates(svm_res)

save(stack, file = "model_info/stack.rda")

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)


# Save blended model stack for reproducibility & easy reference (Rmd report)

blend <- stack %>% 
  blend_predictions(penalty = blend_penalty)

save(blend, file = "model_info/blend.rda")

# Explore the blended model stack

autoplot(blend, type = "weights") 

# fit to ensemble to entire training set ----
stack_fit <- blend %>% 
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
save(stack_fit, file = "model_info/wildfires_stack_trained.rda")


# Explore and assess trained ensemble model
wildfires_test <- stack_fit %>% 
  predict(wildfires_test) %>% 
  bind_cols(wildfires_test)

graph <- ggplot(wildfires_test) +
  aes(x = burned, 
      y = .pred) +
  geom_point() + 
  coord_obs_pred()

graph

member_preds <- 
  wildfires_test %>%
  select(burned) %>%
  bind_cols(predict(stack_fit, wildfires_test, members = TRUE))

rmse_ranked <- map_dfr(member_preds, rmse, truth = burned, data = member_preds) %>%
  mutate(member = colnames(member_preds))

save(graph, rmse_ranked, file = "model_info/assessment.rda")






