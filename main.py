import cv2
import numpy as np
from collections import deque
import math
import time

# Initialize video capture
cap = cv2.VideoCapture('C:\\app\\msc finaa\\mainproject-msc-2\\Screen Recording 2025-03-08 132048.mp4')

# Window configuration
window_width = 400
window_height = 300

# Create main control window
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', window_width, 200)
cv2.moveWindow('Control Panel', 0, 0)

# Create neural network visualization window
cv2.namedWindow('Neural Network', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Neural Network', window_width * 2, 400)  # Increased height for better visualization
cv2.moveWindow('Neural Network', window_width, window_height * 2 + 50)

# Create prediction windows
window_names = [
    'Prediction 1 (Base)',
    'Prediction 2',
    'Prediction 3',
    'Prediction 4',
    'Prediction 5',
    'Neural Network Prediction'
]

# Position windows in a grid (2 rows, 3 columns)
for i, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, window_width, window_height)
    
    # Calculate position (3 windows per row)
    row = i // 3
    col = i % 3
    x_pos = col * window_width
    y_pos = 200 + row * window_height
    
    cv2.moveWindow(name, x_pos, y_pos)

# Create trackbars for controlling parameters
cv2.createTrackbar('Frames Ahead', 'Control Panel', 30, 100, lambda x: None)
cv2.createTrackbar('Contour Threshold', 'Control Panel', 25, 50, lambda x: None)
cv2.createTrackbar('Min Contour Area', 'Control Panel', 100, 200, lambda x: None)
cv2.createTrackbar('Learning Rate', 'Control Panel', 5, 10, lambda x: None)  # Learning rate * 0.01

# Parameter sets for each window
parameter_sets = [
    # Window 1 (Base parameters)
    {
        'MIN_CONTOUR_AREA': 100,
        'HISTORY_LENGTH': 5,
        'TRAIL_LENGTH': 20,
        'PREDICTION_COLOR': (0, 255, 255),  # Yellow
        'CURRENT_COLOR': (0, 0, 255),       # Red
        'PREDICTION_MODEL': 'linear',       # Linear model
        'THRESHOLD_OFFSET': 0               # No threshold adjustment
    },
    # Window 2
    {
        'MIN_CONTOUR_AREA': 120,
        'HISTORY_LENGTH': 8,
        'TRAIL_LENGTH': 25,
        'PREDICTION_COLOR': (0, 255, 200),  # Light yellow-green
        'CURRENT_COLOR': (0, 0, 255),       # Red
        'PREDICTION_MODEL': 'acceleration', # Acceleration model
        'THRESHOLD_OFFSET': -5              # Lower threshold (more sensitive)
    },
    # Window 3
    {
        'MIN_CONTOUR_AREA': 80,
        'HISTORY_LENGTH': 10,
        'TRAIL_LENGTH': 30,
        'PREDICTION_COLOR': (0, 200, 255),  # Light orange
        'CURRENT_COLOR': (0, 0, 255),       # Red
        'PREDICTION_MODEL': 'curved',       # Curved model
        'THRESHOLD_OFFSET': 5               # Higher threshold (less sensitive)
    },
    # Window 4
    {
        'MIN_CONTOUR_AREA': 150,
        'HISTORY_LENGTH': 12,
        'TRAIL_LENGTH': 35,
        'PREDICTION_COLOR': (100, 255, 100),  # Light green
        'CURRENT_COLOR': (0, 0, 255),         # Red
        'PREDICTION_MODEL': 'linear',         # Linear model
        'THRESHOLD_OFFSET': 10                # Much higher threshold
    },
    # Window 5
    {
        'MIN_CONTOUR_AREA': 60,
        'HISTORY_LENGTH': 15,
        'TRAIL_LENGTH': 40,
        'PREDICTION_COLOR': (255, 100, 100),  # Light blue
        'CURRENT_COLOR': (0, 0, 255),         # Red
        'PREDICTION_MODEL': 'curved',         # Curved model
        'THRESHOLD_OFFSET': -10               # Much lower threshold
    }
]

# Neural Network Parameters
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        
        # Initialize biases
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        # Store activations for visualization
        self.hidden_activation = np.zeros(hidden_size)
        self.output_activation = np.zeros(output_size)
        
        # Store sizes for visualization
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights for each model (initially equal)
        self.model_weights = np.ones(5) / 5.0
        
        # Store accuracy history for each model
        self.accuracy_history = [deque(maxlen=30) for _ in range(5)]
        self.combined_accuracy_history = deque(maxlen=30)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        # Forward pass through the network
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_activation = self.sigmoid(self.hidden_input)
        
        self.output_input = np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_output
        self.output_activation = self.sigmoid(self.output_input)
        
        return self.output_activation
    
    def update_weights(self, actual_positions, predicted_positions, learning_rate):
        # Calculate errors for each model
        errors = []
        for i in range(5):
            if actual_positions and i < len(predicted_positions) and predicted_positions[i] is not None:
                # Calculate Euclidean distance between prediction and actual position
                error = np.sqrt((predicted_positions[i][0] - actual_positions[0])**2 + 
                               (predicted_positions[i][1] - actual_positions[1])**2)
                
                # Convert to accuracy (lower error = higher accuracy)
                max_error = 100  # Maximum expected error
                accuracy = max(0, 1 - (error / max_error))
                
                # Store accuracy history
                self.accuracy_history[i].append(accuracy)
                errors.append(error)
            else:
                errors.append(float('inf'))
        
        # Update model weights based on accuracy
        if not all(e == float('inf') for e in errors):
            # Convert errors to weights (lower error = higher weight)
            inv_errors = [1/e if e > 0 else 1.0 for e in errors]
            sum_inv_errors = sum(inv_errors)
            
            if sum_inv_errors > 0:
                new_weights = [ie/sum_inv_errors for ie in inv_errors]
                
                # Smooth weight updates
                self.model_weights = (1 - learning_rate) * self.model_weights + learning_rate * np.array(new_weights)
                self.model_weights = self.model_weights / np.sum(self.model_weights)  # Normalize
    
    def predict_combined(self, predictions):
        # Combine predictions using learned weights
        if not predictions or all(p is None for p in predictions):
            return None
        
        # Filter out None predictions
        valid_predictions = [(i, p) for i, p in enumerate(predictions) if p is not None]
        
        if not valid_predictions:
            return None
        
        # Weighted average of predictions
        x_weighted = 0
        y_weighted = 0
        total_weight = 0
        
        for i, pred in valid_predictions:
            weight = self.model_weights[i]
            x_weighted += pred[0] * weight
            y_weighted += pred[1] * weight
            total_weight += weight
        
        if total_weight > 0:
            return (int(x_weighted / total_weight), int(y_weighted / total_weight))
        else:
            # If no valid weights, return average of predictions
            x_avg = sum(p[0] for _, p in valid_predictions) / len(valid_predictions)
            y_avg = sum(p[1] for _, p in valid_predictions) / len(valid_predictions)
            return (int(x_avg), int(y_avg))
    
    def visualize(self, canvas_width, canvas_height):
        # Create a blank canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Set background color (dark gray)
        canvas[:, :] = (30, 30, 30)
        
        # Define node positions
        input_layer_x = 100
        hidden_layer_x = canvas_width // 2
        output_layer_x = canvas_width - 100
        
        # Calculate vertical spacing for nodes
        input_spacing = canvas_height // (self.input_size + 1)
        hidden_spacing = canvas_height // (self.hidden_size + 1)
        output_spacing = canvas_height // (self.output_size + 1)
        
        # Store node positions for drawing connections
        input_positions = []
        hidden_positions = []
        output_positions = []
        
        # Draw input nodes (pink)
        for i in range(self.input_size):
            y_pos = (i + 1) * input_spacing
            input_positions.append((input_layer_x, y_pos))
            
            # Use model weights as input activation
            activation = self.model_weights[i] if i < len(self.model_weights) else 0
            
            # Draw node
            cv2.circle(canvas, (input_layer_x, y_pos), 25, (180, 105, 255), -1)  # Pink fill
            cv2.circle(canvas, (input_layer_x, y_pos), 25, (255, 255, 255), 1)   # White border
            
            # Add model name
            if i < 5:
                model_name = parameter_sets[i]['PREDICTION_MODEL']
                cv2.putText(canvas, model_name, (input_layer_x - 30, y_pos - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add weight value inside node
                weight_text = f"{activation:.2f}"
                cv2.putText(canvas, weight_text, (input_layer_x - 20, y_pos + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw hidden nodes (yellow)
        for i in range(self.hidden_size):
            y_pos = (i + 1) * hidden_spacing
            hidden_positions.append((hidden_layer_x, y_pos))
            
            # Fix: Safely extract scalar value from hidden activation
            try:
                if isinstance(self.hidden_activation[i], np.ndarray):
                    # Get the first element if it's an array
                    activation = self.hidden_activation[i].item(0) if self.hidden_activation[i].size > 0 else 0.0
                else:
                    activation = self.hidden_activation[i]
            except (ValueError, TypeError, IndexError):
                activation = 0.0
            
            # Draw node
            cv2.circle(canvas, (hidden_layer_x, y_pos), 20, (0, 215, 255), -1)  # Yellow fill
            cv2.circle(canvas, (hidden_layer_x, y_pos), 20, (255, 255, 255), 1)  # White border
            
            # Add activation value inside node
            act_text = f"{activation:.2f}"
            cv2.putText(canvas, act_text, (hidden_layer_x - 20, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw output node (blue)
        for i in range(self.output_size):
            y_pos = (i + 1) * output_spacing
            output_positions.append((output_layer_x, y_pos))
            
            # Fix: Safely extract scalar value from output activation
            try:
                if isinstance(self.output_activation[i], np.ndarray):
                    # Get the first element if it's an array
                    activation = self.output_activation[i].item(0) if self.output_activation[i].size > 0 else 0.0
                else:
                    activation = self.output_activation[i]
            except (ValueError, TypeError, IndexError):
                activation = 0.0
            
            # Draw node
            cv2.circle(canvas, (output_layer_x, y_pos), 25, (255, 128, 0), -1)  # Blue fill
            cv2.circle(canvas, (output_layer_x, y_pos), 25, (255, 255, 255), 1)  # White border
            
            # Add output name
            output_names = ["X Position", "Y Position"]
            if i < len(output_names):
                cv2.putText(canvas, output_names[i], (output_layer_x - 40, y_pos - 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add activation value inside node
                act_text = f"{activation:.2f}"
                cv2.putText(canvas, act_text, (output_layer_x - 20, y_pos + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw connections between input and hidden layer with arrows
        for i, input_pos in enumerate(input_positions):
            for j, hidden_pos in enumerate(hidden_positions):
                # Weight determines line thickness and color
                weight = self.weights_input_hidden[i, j]
                thickness = max(1, min(3, int(abs(weight) * 5)))
                
                if weight >= 0:
                    color = (0, int(min(255, weight * 200)), 0)  # Green for positive
                else:
                    color = (0, 0, int(min(255, abs(weight) * 200)))  # Red for negative
                
                # Draw arrow line
                self.draw_arrow(canvas, input_pos, hidden_pos, color, thickness)
                
                # Add weight text at midpoint
                mid_x = (input_pos[0] + hidden_pos[0]) // 2
                mid_y = (input_pos[1] + hidden_pos[1]) // 2
                
                # Offset text position to not overlap with line
                offset_x = 5 if hidden_pos[1] > input_pos[1] else -25
                offset_y = 5 if hidden_pos[0] > input_pos[0] else -5
                
                # Only show weight text for significant weights
                if abs(weight) > 0.05:
                    cv2.putText(canvas, f"{weight:.2f}", (mid_x + offset_x, mid_y + offset_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw connections between hidden and output layer with arrows
        for i, hidden_pos in enumerate(hidden_positions):
            for j, output_pos in enumerate(output_positions):
                # Weight determines line thickness and color
                weight = self.weights_hidden_output[i, j]
                thickness = max(1, min(3, int(abs(weight) * 5)))
                
                if weight >= 0:
                    color = (0, int(min(255, weight * 200)), 0)  # Green for positive
                else:
                    color = (0, 0, int(min(255, abs(weight) * 200)))  # Red for negative
                
                # Draw arrow line
                self.draw_arrow(canvas, hidden_pos, output_pos, color, thickness)
                
                # Add weight text at midpoint
                mid_x = (hidden_pos[0] + output_pos[0]) // 2
                mid_y = (hidden_pos[1] + output_pos[1]) // 2
                
                # Offset text position to not overlap with line
                offset_x = 5 if output_pos[1] > hidden_pos[1] else -25
                offset_y = 5 if output_pos[0] > hidden_pos[0] else -5
                
                # Only show weight text for significant weights
                if abs(weight) > 0.05:
                    cv2.putText(canvas, f"{weight:.2f}", (mid_x + offset_x, mid_y + offset_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw accuracy graph
        graph_height = 100
        graph_width = canvas_width - 100
        graph_x = 50
        graph_y = canvas_height - 50
        
        # Draw graph background
        cv2.rectangle(canvas, (graph_x, graph_y - graph_height), 
                     (graph_x + graph_width, graph_y), (50, 50, 50), -1)
        
        # Draw grid lines
        for i in range(0, 101, 20):
            y_pos = graph_y - int(i * graph_height / 100)
            cv2.line(canvas, (graph_x, y_pos), (graph_x + graph_width, y_pos), 
                    (100, 100, 100), 1)
            cv2.putText(canvas, f"{i}%", (graph_x - 30, y_pos + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw accuracy lines for each model
        colors = [(0, 255, 255), (0, 255, 200), (0, 200, 255), 
                 (100, 255, 100), (255, 100, 100), (255, 255, 255)]
        
        for i in range(5):
            if self.accuracy_history[i]:
                points = []
                for j, acc in enumerate(self.accuracy_history[i]):
                    x = graph_x + j * graph_width // 30
                    y = graph_y - int(acc * graph_height)
                    points.append((x, y))
                
                for j in range(1, len(points)):
                    cv2.line(canvas, points[j-1], points[j], colors[i], 2)
                
                # Add model label
                model_name = parameter_sets[i]['PREDICTION_MODEL']
                label_x = graph_x + graph_width + 10
                label_y = graph_y - graph_height + i * 20
                cv2.putText(canvas, model_name, (label_x, label_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
        
        # Draw combined accuracy if available
        if self.combined_accuracy_history:
            points = []
            for j, acc in enumerate(self.combined_accuracy_history):
                x = graph_x + j * graph_width // 30
                y = graph_y - int(acc * graph_height)
                points.append((x, y))
            
            for j in range(1, len(points)):
                cv2.line(canvas, points[j-1], points[j], colors[5], 2)
            
            # Add combined label
            label_x = graph_x + graph_width + 10
            label_y = graph_y - graph_height + 5 * 20
            cv2.putText(canvas, "Combined", (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[5], 1)
        
        # Add title
        cv2.putText(canvas, "Neural Network Visualization", (canvas_width//2 - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return canvas
    
    def draw_arrow(self, img, pt1, pt2, color, thickness=1, arrow_size=10):
        """Draw an arrow from pt1 to pt2 with arrowhead"""
        # Draw the main line
        cv2.line(img, pt1, pt2, color, thickness)
        
        # Calculate arrowhead
        angle = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
        
        # Adjust arrow endpoint to stop before the node
        node_radius = 20  # Approximate radius of destination node
        end_x = int(pt2[0] - node_radius * math.cos(angle))
        end_y = int(pt2[1] - node_radius * math.sin(angle))
        
        # Draw arrowhead
        arrow_pt1_x = end_x - arrow_size * math.cos(angle - math.pi/6)
        arrow_pt1_y = end_y - arrow_size * math.sin(angle - math.pi/6)
        arrow_pt2_x = end_x - arrow_size * math.cos(angle + math.pi/6)
        arrow_pt2_y = end_y - arrow_size * math.sin(angle + math.pi/6)
        
        cv2.line(img, (end_x, end_y), (int(arrow_pt1_x), int(arrow_pt1_y)), color, thickness)
        cv2.line(img, (end_x, end_y), (int(arrow_pt2_x), int(arrow_pt2_y)), color, thickness)

# Create neural network (5 inputs for each model, 4 hidden neurons, 2 outputs for x,y)
neural_net = NeuralNetwork(5, 4, 2)  # Reduced hidden neurons for clearer visualization

class PredictionModel:
    def __init__(self, name):
        self.name = name
        
    def predict(self, obj, frames_ahead):
        pass

class LinearModel(PredictionModel):
    def __init__(self):
        super().__init__("Linear")
    
    def predict(self, obj, frames_ahead):
        dx = obj.velocity[0] * frames_ahead
        dy = obj.velocity[1] * frames_ahead
        future_pos = (int(obj.positions[-1][0] + dx),
                      int(obj.positions[-1][1] + dy))
        return future_pos, obj.velocity

class AccelerationModel(PredictionModel):
    def __init__(self):
        super().__init__("Acceleration")
    
    def predict(self, obj, frames_ahead):
        if len(obj.velocities) < 2:
            # Fall back to linear if not enough velocity history
            dx = obj.velocity[0] * frames_ahead
            dy = obj.velocity[1] * frames_ahead
        else:
            # Calculate acceleration
            accel_x = obj.velocities[-1][0] - obj.velocities[-2][0]
            accel_y = obj.velocities[-1][1] - obj.velocities[-2][1]
            
            # Apply kinematic equation: s = ut + 0.5at²
            dx = obj.velocity[0] * frames_ahead + 0.5 * accel_x * frames_ahead * frames_ahead
            dy = obj.velocity[1] * frames_ahead + 0.5 * accel_y * frames_ahead * frames_ahead
        
        future_pos = (int(obj.positions[-1][0] + dx),
                      int(obj.positions[-1][1] + dy))
        return future_pos, (dx/frames_ahead, dy/frames_ahead)

class CurvedTrajectoryModel(PredictionModel):
    def __init__(self):
        super().__init__("Curved")
    
    def predict(self, obj, frames_ahead):
        if len(obj.positions) < 3:
            # Fall back to linear if not enough position history
            dx = obj.velocity[0] * frames_ahead
            dy = obj.velocity[1] * frames_ahead
            future_pos = (int(obj.positions[-1][0] + dx),
                          int(obj.positions[-1][1] + dy))
            return future_pos, obj.velocity
        
        # Fit a quadratic curve to recent positions
        positions = list(obj.positions)
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        # Time points (assuming equal time intervals)
        t = np.arange(len(positions))
        
        # Fit quadratic polynomials
        x_poly = np.polyfit(t, x_coords, 2)
        y_poly = np.polyfit(t, y_coords, 2)
        
        # Predict future position
        future_t = len(positions) - 1 + frames_ahead
        future_x = np.polyval(x_poly, future_t)
        future_y = np.polyval(y_poly, future_t)
        
        # Calculate velocity at prediction point
        dx_dt = 2 * x_poly[0] * future_t + x_poly[1]
        dy_dt = 2 * y_poly[0] * future_t + y_poly[1]
        
        return (int(future_x), int(future_y)), (dx_dt, dy_dt)

class TrackedObject:
    def __init__(self, obj_id, position, contour, history_length, trail_length):
        self.id = obj_id
        self.positions = deque([position], maxlen=history_length)
        self.contours = deque([contour], maxlen=history_length)
        self.velocity = (0, 0)
        self.velocities = deque([(0, 0)], maxlen=history_length)
        self.trail = deque(maxlen=trail_length)
        self.prediction_models = {
            "linear": LinearModel(),
            "acceleration": AccelerationModel(),
            "curved": CurvedTrajectoryModel()
        }
        # Store past predictions for evaluation
        self.past_predictions = {}  # {frame_number: [(model_idx, prediction), ...]}
        
    def update(self, new_position, new_contour, frame_number):
        # Check past predictions for this frame
        if frame_number in self.past_predictions:
            predictions = self.past_predictions[frame_number]
            actual_position = new_position
            
            # Update neural network weights based on prediction accuracy
            neural_net.update_weights(actual_position, 
                                     [pred for _, pred in sorted(predictions, key=lambda x: x[0])],
                                     learning_rate)
            
            # Remove evaluated predictions
            del self.past_predictions[frame_number]
        
        self.positions.append(new_position)
        self.contours.append(new_contour)
        self.trail.append(new_position)
        self._calculate_velocity()
        
    def _calculate_velocity(self):
        if len(self.positions) >= 2:
            # Weighted average of recent velocities (more weight to recent)
            weights = np.linspace(0.5, 1.0, min(5, len(self.positions)-1))
            weights = weights / np.sum(weights)
            
            dx_values = []
            dy_values = []
            
            for i in range(1, min(6, len(self.positions))):
                dx = self.positions[-1][0] - self.positions[-1-i][0]
                dy = self.positions[-1][1] - self.positions[-1-i][1]
                dx_values.append(dx)
                dy_values.append(dy)
            
            if dx_values:
                dx = np.sum(np.array(dx_values[:len(weights)]) * weights)
                dy = np.sum(np.array(dy_values[:len(weights)]) * weights)
                self.velocity = (dx, dy)
                self.velocities.append(self.velocity)
    
    def predict(self, frames_ahead, model_name='linear'):
        # Use the specified prediction model
        return self.prediction_models[model_name].predict(self, frames_ahead)
    
    def store_prediction(self, model_idx, prediction, future_frame):
        # Store prediction for future evaluation
        if future_frame not in self.past_predictions:
            self.past_predictions[future_frame] = []
        
        self.past_predictions[future_frame].append((model_idx, prediction))

def draw_curved_path(img, points, color, thickness=1):
    if len(points) < 3:
        for i in range(1, len(points)):
            cv2.line(img, points[i-1], points[i], color, thickness)
        return
    
    # Convert points to numpy array
    points = np.array(points, dtype=np.int32)
    
    # Fit a spline curve
    t = np.arange(len(points))
    x = points[:, 0]
    y = points[:, 1]
    
    # Interpolate with more points for smoothness
    t_interp = np.linspace(0, len(points)-1, len(points)*5)
    x_interp = np.interp(t_interp, t, x)
    y_interp = np.interp(t_interp, t, y)
    
    # Draw the smooth curve
    points_interp = np.column_stack((x_interp, y_interp)).astype(np.int32)
    for i in range(1, len(points_interp)):
        cv2.line(img, tuple(points_interp[i-1]), tuple(points_interp[i]), color, thickness)

# Create separate tracking objects for each window
tracked_objects_sets = [{} for _ in range(len(parameter_sets))]
current_ids = [0 for _ in range(len(parameter_sets))]

# Create a set for neural network prediction objects
nn_tracked_objects = {}
nn_current_id = 0

prev_gray = None
frame_number = 0
start_time = time.time()
learning_rate = 0.05  # Default learning rate

# Resize outputs while maintaining aspect ratio
def resize_aspect(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        ratio = height / h
        dim = (int(w * ratio), height)
    else:
        ratio = width / w
        dim = (width, int(h * ratio))
    return cv2.resize(img, dim)

# Store predictions from all models
all_predictions = {}  # {obj_id: [pred1, pred2, ...]}

while True:
    ret, frame = cap.read()
    if not ret:
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_number = 0
        continue
    
    # Get parameters from trackbars
    prediction_frames = cv2.getTrackbarPos('Frames Ahead', 'Control Panel')
    contour_threshold = cv2.getTrackbarPos('Contour Threshold', 'Control Panel')
    base_min_contour_area = cv2.getTrackbarPos('Min Contour Area', 'Control Panel')
    learning_rate_value = cv2.getTrackbarPos('Learning Rate', 'Control Panel') / 100.0
    learning_rate = learning_rate_value
    
    # Preprocess frame
    original_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    if prev_gray is None:
        prev_gray = gray.copy()
        continue
    
    # Process each window with its own parameters
    for window_idx, params in enumerate(parameter_sets):
        # Apply parameter adjustments
        MIN_CONTOUR_AREA = base_min_contour_area + params['MIN_CONTOUR_AREA'] - 100  # Adjust based on base value
        HISTORY_LENGTH = params['HISTORY_LENGTH']
        TRAIL_LENGTH = params['TRAIL_LENGTH']
        PREDICTION_COLOR = params['PREDICTION_COLOR']
        CURRENT_COLOR = params['CURRENT_COLOR']
        PREDICTION_MODEL = params['PREDICTION_MODEL']
        THRESHOLD_OFFSET = params['THRESHOLD_OFFSET']
        
        # Window-specific motion detection with adjusted threshold
        diff = cv2.absdiff(gray, prev_gray)
        adjusted_threshold = max(5, min(50, contour_threshold + THRESHOLD_OFFSET))
        _, thresh = cv2.threshold(diff, adjusted_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tracked_objects = tracked_objects_sets[window_idx]
        current_id = current_ids[window_idx]
        
        current_objects = []
        future_motion_frame = original_frame.copy()

        # Process contours
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w//2
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h//2
            current_objects.append((cX, cY, cnt))

        # Object tracking
        active_ids = set()
        for cX, cY, cnt in current_objects:
            min_dist = float('inf')
            best_obj = None
            
            for obj_id, obj in tracked_objects.items():
                last_pos = obj.positions[-1]
                distance = np.sqrt((cX - last_pos[0])**2 + (cY - last_pos[1])**2)
                
                if distance < min_dist and distance < 50:
                    min_dist = distance
                    best_obj = obj
                    
            if best_obj:
                best_obj.update((cX, cY), cnt, frame_number)
                active_ids.add(best_obj.id)
            else:
                new_obj = TrackedObject(current_id, (cX, cY), cnt, HISTORY_LENGTH, TRAIL_LENGTH)
                tracked_objects[current_id] = new_obj
                active_ids.add(current_id)
                current_id += 1

        # Remove stale objects
        for obj_id in list(tracked_objects.keys()):
            if obj_id not in active_ids:
                del tracked_objects[obj_id]
        
        # Update current_id for this window
        current_ids[window_idx] = current_id
        
        # Prediction and visualization
        for obj_id, obj in tracked_objects.items():
            # Draw current state
            cv2.drawContours(future_motion_frame, [obj.contours[-1]], -1, CURRENT_COLOR, 1)
            cv2.circle(future_motion_frame, obj.positions[-1], 3, CURRENT_COLOR, -1)
            
            # Calculate prediction using the specified model
            future_pos, future_velocity = obj.predict(prediction_frames, PREDICTION_MODEL)
            
            # Store prediction for neural network evaluation
            future_frame = frame_number + prediction_frames
            obj.store_prediction(window_idx, future_pos, future_frame)
            
            # Store prediction for combined display
            if obj_id not in all_predictions:
                all_predictions[obj_id] = [None] * 5
            all_predictions[obj_id][window_idx] = future_pos
            
            # Warp contour for prediction
            dx = future_pos[0] - obj.positions[-1][0]
            dy = future_pos[1] - obj.positions[-1][1]
            warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            predicted_contour = cv2.transform(obj.contours[-1], warp_matrix)
            
            # Draw prediction
            cv2.drawContours(future_motion_frame, [predicted_contour.astype(np.int32)], 
                            -1, PREDICTION_COLOR, 1)
            cv2.circle(future_motion_frame, future_pos, 3, PREDICTION_COLOR, -1)
            
            # Draw predicted path (curved or straight depending on model)
            predicted_path = [obj.positions[-1]]
            for i in range(1, prediction_frames + 1, max(1, prediction_frames // 5)):
                intermediate_pos, _ = obj.predict(i, PREDICTION_MODEL)
                predicted_path.append(intermediate_pos)
            
            if PREDICTION_MODEL == 'curved' and len(predicted_path) >= 3:
                draw_curved_path(future_motion_frame, predicted_path, PREDICTION_COLOR, 1)
            else:
                for i in range(1, len(predicted_path)):
                    cv2.line(future_motion_frame, predicted_path[i-1], predicted_path[i], PREDICTION_COLOR, 1)
            
            # Draw motion trail
            if len(obj.trail) > 1:
                trail_points = list(obj.trail)
                for i in range(1, len(trail_points)):
                    color_start = (0, 0, 255)  # Red
                    color_end = (255, 0, 0)    # Blue
                    color = tuple(map(int, np.linspace(color_start, color_end, len(trail_points))[i-1]))
                    cv2.line(future_motion_frame, 
                            trail_points[i-1], trail_points[i],
                            color, 1)

        # Add parameter text to the frame
        param_text = f"Area: {MIN_CONTOUR_AREA}, Thresh: {adjusted_threshold}, Model: {PREDICTION_MODEL}"
        cv2.putText(future_motion_frame, param_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize and display
        resized_future = resize_aspect(future_motion_frame, width=window_width)
        cv2.imshow(window_names[window_idx], resized_future)
    
    # Process neural network prediction (6th window)
    nn_prediction_frame = original_frame.copy()
    
    # Only start neural network prediction after 60 frames (2 seconds)
    if frame_number >= 60:
        # Process each object with neural network
        for obj_id, predictions in all_predictions.items():
            # Forward pass through neural network
            input_vector = []
            for pred in predictions:
                if pred is not None:
                    # Normalize coordinates to [0,1] range for neural network
                    h, w = original_frame.shape[:2]
                    x_norm = pred[0] / w
                    y_norm = pred[1] / h
                    input_vector.extend([x_norm, y_norm])
                else:
                    input_vector.extend([0, 0])
            
            # Ensure input vector has correct length
            if len(input_vector) > 0:
                # Pad or truncate to match input size
                input_vector = np.array(input_vector[:neural_net.input_size]).reshape(1, -1)
                
                # Forward pass
                output = neural_net.forward(input_vector)
                
                # Denormalize output
                x_pred = int(output[0, 0] * w)
                y_pred = int(output[0, 1] * h)
                nn_prediction = (x_pred, y_pred)
                
                # Get neural network prediction
                combined_pred = neural_net.predict_combined(predictions)
                
                if combined_pred:
                    # Find the object in one of the windows to get contour
                    obj = None
                    for window_idx in range(len(parameter_sets)):
                        if obj_id in tracked_objects_sets[window_idx]:
                            obj = tracked_objects_sets[window_idx][obj_id]
                            break
                    
                    if obj:
                        # Draw current position
                        cv2.drawContours(nn_prediction_frame, [obj.contours[-1]], -1, (0, 0, 255), 1)
                        cv2.circle(nn_prediction_frame, obj.positions[-1], 3, (0, 0, 255), -1)
                        
                        # Warp contour for prediction
                        dx = combined_pred[0] - obj.positions[-1][0]
                        dy = combined_pred[1] - obj.positions[-1][1]
                        warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                        predicted_contour = cv2.transform(obj.contours[-1], warp_matrix)
                        
                        # Draw neural network prediction
                        cv2.drawContours(nn_prediction_frame, [predicted_contour.astype(np.int32)], 
                                        -1, (255, 255, 0), 1)  # Bright yellow
                        cv2.circle(nn_prediction_frame, combined_pred, 3, (255, 255, 0), -1)
                        
                        # Draw line from current to predicted position
                        cv2.line(nn_prediction_frame, obj.positions[-1], combined_pred, (255, 255, 0), 1)
                        
                        # Draw motion trail
                        if len(obj.trail) > 1:
                            trail_points = list(obj.trail)
                            for i in range(1, len(trail_points)):
                                color_start = (0, 0, 255)  # Red
                                color_end = (255, 0, 0)    # Blue
                                color = tuple(map(int, np.linspace(color_start, color_end, len(trail_points))[i-1]))
                                cv2.line(nn_prediction_frame, 
                                        trail_points[i-1], trail_points[i],
                                        color, 1)
                        
                        # Calculate accuracy if we have past predictions to compare
                        if obj.past_predictions:
                            # Find the oldest prediction to evaluate
                            oldest_frame = min(obj.past_predictions.keys())
                            if oldest_frame <= frame_number:
                                predictions = obj.past_predictions[oldest_frame]
                                actual_position = obj.positions[-1]
                                
                                # Get combined prediction
                                model_predictions = [None] * 5
                                for model_idx, pred in predictions:
                                    model_predictions[model_idx] = pred
                                
                                combined_pred = neural_net.predict_combined(model_predictions)
                                
                                if combined_pred:
                                    # Calculate accuracy
                                    error = np.sqrt((combined_pred[0] - actual_position[0])**2 + 
                                                  (combined_pred[1] - actual_position[1])**2)
                                    max_error = 100
                                    accuracy = max(0, 1 - (error / max_error))
                                    
                                    # Store combined accuracy
                                    neural_net.combined_accuracy_history.append(accuracy)
    else:
        # Display waiting message
        cv2.putText(nn_prediction_frame, "Collecting data for neural network...", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(nn_prediction_frame, f"Starting in {60-frame_number} frames", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Add neural network weights to the frame
    y_offset = 20
    cv2.putText(nn_prediction_frame, "Neural Network Weights:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    
    for i, weight in enumerate(neural_net.model_weights):
        model_name = parameter_sets[i]['PREDICTION_MODEL']
        cv2.putText(nn_prediction_frame, f"{model_name}: {weight:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Display neural network prediction
    resized_nn = resize_aspect(nn_prediction_frame, width=window_width)
    cv2.imshow(window_names[5], resized_nn)
    
    # Create and display neural network visualization
    nn_vis = neural_net.visualize(window_width * 2, 400)
    cv2.imshow('Neural Network', nn_vis)
    
    prev_gray = gray.copy()
    frame_number += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
