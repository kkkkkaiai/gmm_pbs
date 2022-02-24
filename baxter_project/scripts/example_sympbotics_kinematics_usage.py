import baxter_left_kinematics as blk

#get the jacobian for the 1st joint with all joint angles equal to zero
blk.J[0]([0]*7)

#get the jacobian for the 7th joint with all joint angles equal to 0.707 radians
blk.J[6]([0.707]*7)

#get forward kinematics for the 1st joint frame with all joint angles equal to zero
blk.FK[0]([0]*7)

#get forward kinematics for the 7th joint frame with all joint angles equal to zero
blk.FK[6]([0]*7)

        
