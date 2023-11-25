python
from deepface import DeepFace

class FaceAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
    
    def analyze_face(self):
        demography = DeepFace.analyze(self.image_path, actions=['age', 'gender', 'race', 'emotion'])
        return demography

image_path = "juan.jpg"
analyzer = FaceAnalyzer(image_path)
demography = analyzer.analyze_face()

print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])
