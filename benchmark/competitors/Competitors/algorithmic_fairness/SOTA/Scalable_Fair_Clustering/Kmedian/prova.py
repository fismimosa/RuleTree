import matlab.engine
eng = matlab.engine.start_matlab()
print("MATLAB started successfully")
eng.quit()