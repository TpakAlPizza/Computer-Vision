# number_detector_game.py
import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time

class NumberGame:
    def __init__(self):
        print("🎮 Welcome to the Number Guessing Game!")
        print("=" * 60)
        
        # Game settings
        self.canvas_size = 400
        self.brush_size = 18
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.drawing = False
        
        # Game stats
        self.score = 0
        self.total_guesses = 0
        self.correct_guesses = 0
        self.learning_mode = False
        
        # Load or create model
        self.model, self.training_data = self.load_or_create_model()
        
        # Current game state
        self.current_prediction = None
        self.current_confidence = 0
        self.target_number = None
    
    def load_or_create_model(self):
        model_file = 'number_game_model.pkl'
        data_file = 'training_data.pkl'
        
        if os.path.exists(model_file) and os.path.exists(data_file):
            print("Loading existing game model...")
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(data_file, 'rb') as f:
                training_data = pickle.load(f)
            print(f"Loaded model with {len(training_data['X'])} training examples")
        else:
            print("Creating new game model...")
            model, training_data = self.create_initial_model()
        
        return model, training_data
    
    def create_initial_model(self):

        from sklearn.datasets import load_digits
        
        # Start with sklearn digits
        digits = load_digits()
        X = digits.images.reshape((len(digits.images), -1))
        y = digits.target
        
        # Train KNN model
        model = KNeighborsClassifier(n_neighbors=3, weights='distance')
        model.fit(X, y)
        
        # Save initial data
        training_data = {'X': X.tolist(), 'y': y.tolist()}
        
        # Save files
        with open('number_game_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('training_data.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        
        print("Created initial model with sklearn digits")
        return model, training_data
    
    def preprocess_drawing(self):
        
        if np.sum(self.canvas) == 0:
            return None
        
        # Find drawn area
        points = np.where(self.canvas > 10)
        if len(points[0]) == 0:
            return None
        
        # Get bounding box with padding
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        
        padding = 30
        y_min = max(0, y_min - padding)
        y_max = min(self.canvas_size, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(self.canvas_size, x_max + padding)
        
        digit = self.canvas[y_min:y_max, x_min:x_max]
        
        if digit.shape[0] == 0 or digit.shape[1] == 0:
            return None
        
        # Resize to 8x8
        resized = cv2.resize(digit, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Normalize
        normalized = (resized / 255.0) * 16.0
        
        return normalized.flatten()
    
    def make_prediction(self):
        
        processed = self.preprocess_drawing()
        if processed is None:
            return None, 0
        
        # Reshape for prediction
        X_pred = processed.reshape(1, -1)
        
        # Get prediction and confidence
        pred = self.model.predict(X_pred)[0]
        proba = self.model.predict_proba(X_pred)[0]
        confidence = proba[pred]
        
        return pred, confidence
    
    def add_training_example(self, drawing, correct_label):
        
        processed = self.preprocess_drawing()
        if processed is None:
            return
        
        # Add to training data
        self.training_data['X'].append(processed.tolist())
        self.training_data['y'].append(correct_label)
        
        # Convert to numpy arrays
        X = np.array(self.training_data['X'])
        y = np.array(self.training_data['y'])
        
        # Retrain model
        print(f"Training with {len(X)} examples...")
        self.model.fit(X, y)
        
        # Save updated model and data
        with open('number_game_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('training_data.pkl', 'wb') as f:
            pickle.dump(self.training_data, f)
        
        print(f"✓ Learned example of number {correct_label} (now {len(X)} total examples)")
    
    def generate_target_number(self):
        import random
        self.target_number = random.randint(0, 9)
        return self.target_number
    
    def check_guess(self, guess):
    
        if guess == self.target_number:
            self.score += 1
            self.correct_guesses += 1
            return True
        return False
    
    def show_game_stats(self):

        accuracy = (self.correct_guesses / self.total_guesses * 100) if self.total_guesses > 0 else 0
        
        print("\n" + "="*50)
        print("📊 GAME STATISTICS")
        print("="*50)
        print(f"🏆 Score: {self.score}")
        print(f"🎯 Accuracy: {accuracy:.1f}%")
        print(f"📈 Total Guesses: {self.total_guesses}")
        print(f"✅ Correct: {self.correct_guesses}")
        print(f"📚 Training Examples: {len(self.training_data['X'])}")
        print("="*50)
    
    def run_game_mode(self):
        print("\n" + "="*60)
        print("🎮 GAME MODE: Can the computer guess your number?")
        print("="*60)
        print("\nRULES:")
        print("1. Draw any number (0-9)")
        print("2. Computer will guess")
        print("3. Press Y if correct, N if wrong")
        print("4. If wrong, press 0-9 to teach the correct number")
        print("\nCONTROLS:")
        print("Y = Correct guess      N = Wrong guess")
        print("C = Clear canvas       Q = Quit game")
        print("S = Show stats         R = Retry same number")
        print("="*60 + "\n")
        
        def draw_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                cv2.circle(self.canvas, (x, y), self.brush_size, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                cv2.circle(self.canvas, (x, y), self.brush_size, 255, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
        
        cv2.namedWindow("Number Game - Draw Here", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Number Game - Draw Here", 500, 500)
        cv2.setMouseCallback("Number Game - Draw Here", draw_callback)
        
        last_prediction = None
        
        while True:
            # Create display
            display = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
            
            # Draw center guide
            center = self.canvas_size // 2
            cv2.circle(display, (center, center), 5, (0, 0, 255), -1)
            
            # Make prediction
            pred, confidence = self.make_prediction()
            
            # Update display if new prediction
            if pred is not None and pred != last_prediction:
                last_prediction = pred
                self.current_prediction = pred
                self.current_confidence = confidence
                
                self.total_guesses += 1
                
                print(f"\n🤖 Computer guesses: {pred} ({confidence:.1%} confident)")
                print("   Press Y if correct, N if wrong")
            
            # Show prediction on image
            if pred is not None:
                color = (0, 200, 0) if confidence > 0.6 else (0, 165, 255) if confidence > 0.3 else (0, 0, 255)
                cv2.putText(display, f"Guess: {pred}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(display, f"Confidence: {confidence:.0%}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Number Game - Draw Here", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nThanks for playing! 👋")
                self.show_game_stats()
                break
            
            elif key == ord('c'):
                self.canvas[:] = 0
                last_prediction = None
                print("\n🗑️ Canvas cleared! Draw a new number...")
            
            elif key == ord('y') and pred is not None:
                # Computer guessed correctly!
                self.score += 1
                self.correct_guesses += 1
                print(f"\n✅ CORRECT! +1 point!")
                print(f"   Score: {self.score}")
                print(f"   The computer learned from seeing number {pred}")
                
                # Add as positive example
                self.add_training_example(self.canvas, pred)
                
                # Clear for next round
                self.canvas[:] = 0
                last_prediction = None
                time.sleep(1)
                print("\nDraw another number...")
            
            elif key == ord('n') and pred is not None:
                # Computer guessed wrong
                print(f"\n❌ WRONG! What number is this?")
                print("   Press the correct number key (0-9) to teach me")
                print("   Or press C to clear and try again")
            
            elif ord('0') <= key <= ord('9') and pred is not None:
                # User is teaching the correct number
                correct_num = key - ord('0')
                
                print(f"\n📚 TEACHING: You say this is {correct_num}")
                print(f"   Computer guessed {pred} (was {self.current_confidence:.1%} confident)")
                
                # Add as training example
                self.add_training_example(self.canvas, correct_num)
                
                # Clear canvas
                self.canvas[:] = 0
                last_prediction = None
                print("\nThanks for teaching me! Draw another number...")
            
            elif key == ord('s'):
                self.show_game_stats()
            
            elif key == ord('r'):
                print(f"\n🔄 Retrying... Draw the same number again")
                self.canvas[:] = 0
                last_prediction = None
    
    def run_challenge_mode(self):
        
        print("\n" + "="*60)
        print("🎯 CHALLENGE MODE: Can you draw what the computer asks?")
        print("="*60)
        print("\nRULES:")
        print("1. Computer shows a number to draw")
        print("2. You draw it as best you can")
        print("3. Computer tries to recognize it")
        print("4. Score points for correct recognition!")
        print("\nCONTROLS:")
        print("Space = New number    C = Clear canvas")
        print("Q = Quit              S = Show stats")
        print("="*60 + "\n")
        
        # Generate first target
        self.generate_target_number()
        print(f"\n🎯 FIRST CHALLENGE: Draw the number {self.target_number}")
        print("   Make it clear and centered!")
        
        def draw_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                cv2.circle(self.canvas, (x, y), self.brush_size, 255, -1)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                cv2.circle(self.canvas, (x, y), self.brush_size, 255, -1)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
        
        cv2.namedWindow("Challenge Mode", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Challenge Mode", 500, 500)
        cv2.setMouseCallback("Challenge Mode", draw_callback)
        
        game_active = True
        
        while game_active:
            # Create display
            display = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
            
            # Draw center guide
            center = self.canvas_size // 2
            cv2.circle(display, (center, center), 5, (0, 0, 255), -1)
            
            # Show target number
            cv2.putText(display, f"Draw: {self.target_number}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(display, f"Score: {self.score}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            # Make prediction
            pred, confidence = self.make_prediction()
            
            # Show prediction if available
            if pred is not None:
                color = (0, 255, 0) if pred == self.target_number else (0, 0, 255)
                cv2.putText(display, f"Computer sees: {pred}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                if pred == self.target_number:
                    cv2.putText(display, "CORRECT! Press SPACE", (20, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Challenge Mode", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                game_active = False
                print("\nChallenge mode ended!")
                self.show_game_stats()
            
            elif key == ord('c'):
                self.canvas[:] = 0
                print("\nCanvas cleared - try again!")
            
            elif key == ord(' ') and pred is not None:
                # Check if correct
                if pred == self.target_number:
                    self.score += 1
                    self.correct_guesses += 1
                    print(f"\n✅ CORRECT! +1 point!")
                    print(f"   Score: {self.score}")
                    
                    # Add as positive example
                    self.add_training_example(self.canvas, pred)
                else:
                    print(f"\n❌ Wrong! You drew {self.target_number}, computer saw {pred}")
                    print("   Teaching computer the correct answer...")
                    self.add_training_example(self.canvas, self.target_number)
                
                # Clear and get new target
                self.canvas[:] = 0
                self.total_guesses += 1
                self.generate_target_number()
                print(f"\n🎯 NEXT: Draw the number {self.target_number}")
            
            elif key == ord('s'):
                self.show_game_stats()
        
        cv2.destroyAllWindows()
    
    def choose_game_mode(self):
        
        print("\n" + "="*60)
        print("SELECT GAME MODE:")
        print("="*60)
        print("1. 🤖 GUESSING GAME")
        print("   - You draw numbers, computer guesses")
        print("   - Teach computer when it's wrong")
        print("   - Watch computer learn and improve!")
        print()
        print("2. 🎯 CHALLENGE MODE")
        print("   - Computer gives you numbers to draw")
        print("   - Score points for correct recognition")
        print("   - Test your drawing skills!")
        print()
        print("3. 📊 VIEW STATISTICS")
        print("   - See your game history")
        print("   - View computer's learning progress")
        print("="*60)
        
        while True:
            choice = input("\nChoose mode (1, 2, 3) or Q to quit: ").strip().lower()
            
            if choice == '1':
                self.run_game_mode()
                break
            elif choice == '2':
                self.run_challenge_mode()
                break
            elif choice == '3':
                self.show_game_stats()
                self.choose_game_mode()
                break
            elif choice == 'q':
                print("Thanks for playing! 👋")
                return
            else:
                print("Please enter 1, 2, 3, or Q")

def main():
    print("🎨 INTERACTIVE NUMBER DETECTOR GAME")
    print("Learn AI by teaching a computer to recognize numbers!")
    print("\nThe computer starts knowing very little...")
    print("Teach it by showing examples and correcting mistakes!")
    print("Watch as it gets smarter with each game! 🤖")
    
    game = NumberGame()
    game.choose_game_mode()
    
    # Final statistics
    print("\n" + "="*60)
    print("FINAL GAME REPORT")
    print("="*60)
    print(f"🏆 Final Score: {game.score}")
    print(f"📈 Final Accuracy: {(game.correct_guesses/game.total_guesses*100 if game.total_guesses > 0 else 0):.1f}%")
    print(f"📚 Total Training Examples: {len(game.training_data['X'])}")
    print("="*60)
    print("\nCome back and play again to help the computer learn more! 🎯")

if __name__ == "__main__":
    main()