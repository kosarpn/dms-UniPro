from features import MouthAspectRatioAnalyzer
analyzer=MouthAspectRatioAnalyzer(yawn_threshold=0.25,min_yawn_frames=5)
for i in range(10):
    result=analyzer.update(0.5)
    print(f"Frame{i+1}:{result}")
