import pyzed.sl as sl

cam = sl.Camera()
params = sl.InitParameters()
status = cam.open(params)

if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", status)
else:
    print("ZED camera opened successfully")

cam.close()