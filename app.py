from flask import Flask, jsonify
from flask_cors import CORS
from stable_baselines3 import PPO
from performance_monitor import SystemPerformanceEnv

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for frontend access

# Load AI model
model = PPO.load("ai_optimizer")
env = SystemPerformanceEnv()

@app.route('/optimize', methods=['GET'])
def optimize():
    obs = env.reset()
    optimization_steps = []

    for _ in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        step_result = {
            "Action": int(action),
            "CPU": float(obs[0]),
            "Memory": float(obs[1]),
            "Latency": float(obs[2]),
            "Reward": float(reward)
        }
        optimization_steps.append(step_result)

        if done:
            break

    return jsonify({"Optimization Steps": optimization_steps})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

