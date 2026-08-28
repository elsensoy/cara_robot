import torch

def apply_stability_push(env, robot_asset, force_magnitude=50.0):
    """
    Simulates a 'push' by applying a sudden force to Cara's torso.
    force_magnitude: Force in Newtons.
    """
    num_envs = env.num_envs
    # Randomize push direction (X and Y horizontal plane)
    push_direction = torch.randn((num_envs, 3), device=env.device)
    push_direction[:, 2] = 0.0 # Keep the push horizontal
    
    # Normalize and scale
    push_force = torch.nn.functional.normalize(push_direction, dim=1) * force_magnitude
    
    # Apply to the 'torso' link (index 0 usually)
    # This happens in the PhysX buffer
    robot_asset.set_external_force_and_torque(push_force, torch.zeros_like(push_force), indices=None)
    
    print(f"Stability Test: Applied {force_magnitude}N push to Cara.")

# Example trigger inside the simulation loop
# if current_step % 250 == 0:
#     apply_stability_push(env, cara_robot)
