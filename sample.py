GEMINI_KEY="AIzaSyAD74i1mSmhCe7-nuzOZdPkzJUTPo6BRBg"
from smart_road_extractor import SmartRoadDataExtractor 
extractor = SmartRoadDataExtractor(GEMINI_KEY)
sample = extractor.analyze_road_image("img 4.jpg")
print(sample) 
