from math import sin, cos
from offset_util import offset_and_reshape
import numpy as np

pi = np.pi


def joint_fk00(q):
    #
    pose = [0] * 16
    #
    x0 = q[0] + pi / 4
    x1 = cos(x0)
    x2 = sin(x0)
    #
    pose[0] = x1
    pose[1] = 0
    pose[2] = -x2
    pose[3] = 0.069 * x1
    pose[4] = x2
    pose[5] = 0
    pose[6] = x1
    pose[7] = 0.069 * x2
    pose[8] = 0
    pose[9] = -1
    pose[10] = 0
    pose[11] = 0.270350000000000
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk01(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[1])
    x1 = q[0] + pi / 4
    x2 = cos(x1)
    x3 = sin(x1)
    x4 = cos(q[1])
    #
    pose[0] = -x0 * x2
    pose[1] = -x3
    pose[2] = x2 * x4
    pose[3] = 0.069 * x2
    pose[4] = -x0 * x3
    pose[5] = x2
    pose[6] = x3 * x4
    pose[7] = 0.069 * x3
    pose[8] = -x4
    pose[9] = 0
    pose[10] = -x0
    pose[11] = 0.270350000000000
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk02(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[2])
    x1 = q[0] + pi / 4
    x2 = sin(x1)
    x3 = x0 * x2
    x4 = sin(q[1])
    x5 = cos(q[2])
    x6 = cos(x1)
    x7 = x5 * x6
    x8 = cos(q[1])
    x9 = x6 * x8
    x10 = x2 * x5
    x11 = x0 * x6
    x12 = 0.069 * x6
    x13 = 0.069 * x2
    x14 = x4 * x5
    x15 = x2 * x8
    x16 = x5 * x8
    #
    pose[0] = -x3 - x4 * x7
    pose[1] = -x9
    pose[2] = -x10 + x11 * x4
    pose[3] = -x0 * x13 - x12 * x14 + x12 + 0.36435 * x9
    pose[4] = -x10 * x4 + x11
    pose[5] = -x15
    pose[6] = x3 * x4 + x7
    pose[7] = x0 * x12 - x13 * x14 + x13 + 0.36435 * x15
    pose[8] = -x16
    pose[9] = x4
    pose[10] = x0 * x8
    pose[11] = -0.069 * x16 - 0.36435 * x4 + 0.27035
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk03(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[3])
    x1 = cos(q[1])
    x2 = x0 * x1
    x3 = q[0] + pi / 4
    x4 = cos(x3)
    x5 = cos(q[3])
    x6 = sin(q[2])
    x7 = sin(x3)
    x8 = x6 * x7
    x9 = sin(q[1])
    x10 = cos(q[2])
    x11 = x10 * x4
    x12 = -x11 * x9 - x8
    x13 = x10 * x7
    x14 = x4 * x6
    x15 = x1 * x5
    x16 = 0.069 * x4
    x17 = 0.069 * x7
    x18 = 0.36435 * x1
    x19 = x10 * x9
    x20 = -x13 * x9 + x14
    x21 = x1 * x10
    #
    pose[0] = x12 * x5 - x2 * x4
    pose[1] = -x13 + x14 * x9
    pose[2] = x0 * x12 + x15 * x4
    pose[3] = -x16 * x19 + x16 - x17 * x6 + x18 * x4
    pose[4] = -x2 * x7 + x20 * x5
    pose[5] = x11 + x8 * x9
    pose[6] = x0 * x20 + x15 * x7
    pose[7] = x16 * x6 - x17 * x19 + x17 + x18 * x7
    pose[8] = x0 * x9 - x21 * x5
    pose[9] = x1 * x6
    pose[10] = -x0 * x21 - x5 * x9
    pose[11] = -0.069 * x21 - 0.36435 * x9 + 0.27035
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk04(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[4])
    x1 = cos(q[2])
    x2 = q[0] + pi / 4
    x3 = sin(x2)
    x4 = x1 * x3
    x5 = sin(q[1])
    x6 = sin(q[2])
    x7 = cos(x2)
    x8 = x6 * x7
    x9 = -x4 + x5 * x8
    x10 = x0 * x9
    x11 = cos(q[4])
    x12 = sin(q[3])
    x13 = cos(q[1])
    x14 = x12 * x13
    x15 = cos(q[3])
    x16 = x3 * x6
    x17 = x1 * x7
    x18 = -x16 - x17 * x5
    x19 = -x14 * x7 + x15 * x18
    x20 = x11 * x19
    x21 = x13 * x15
    x22 = x21 * x7
    x23 = x12 * x18
    x24 = 0.069 * x7
    x25 = 0.069 * x3
    x26 = 0.36435 * x13
    x27 = x1 * x5
    x28 = x16 * x5 + x17
    x29 = x0 * x28
    x30 = -x4 * x5 + x8
    x31 = -x14 * x3 + x15 * x30
    x32 = x11 * x31
    x33 = x21 * x3
    x34 = x12 * x30
    x35 = x13 * x6
    x36 = x0 * x35
    x37 = x1 * x13
    x38 = x12 * x5 - x15 * x37
    x39 = x11 * x38
    x40 = x15 * x5
    x41 = x12 * x37
    #
    pose[0] = x10 + x20
    pose[1] = -x22 - x23
    pose[2] = -x0 * x19 + x11 * x9
    pose[3] = 0.01 * x10 + 0.01 * x20 + 0.37429 * x22 + 0.37429 * x23 - x24 * x27 + x24 - x25 * x6 + x26 * x7
    pose[4] = x29 + x32
    pose[5] = -x33 - x34
    pose[6] = -x0 * x31 + x11 * x28
    pose[7] = x24 * x6 - x25 * x27 + x25 + x26 * x3 + 0.01 * x29 + 0.01 * x32 + 0.37429 * x33 + 0.37429 * x34
    pose[8] = x36 + x39
    pose[9] = x40 + x41
    pose[10] = -x0 * x38 + x11 * x35
    pose[11] = 0.01 * x36 - 0.069 * x37 + 0.01 * x39 - 0.37429 * x40 - 0.37429 * x41 - 0.36435 * x5 + 0.27035
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk05(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[5])
    x1 = cos(q[1])
    x2 = cos(q[3])
    x3 = x1 * x2
    x4 = q[0] + pi / 4
    x5 = cos(x4)
    x6 = x3 * x5
    x7 = sin(q[3])
    x8 = sin(q[2])
    x9 = sin(x4)
    x10 = x8 * x9
    x11 = sin(q[1])
    x12 = cos(q[2])
    x13 = x12 * x5
    x14 = -x10 - x11 * x13
    x15 = x14 * x7
    x16 = -x15 - x6
    x17 = cos(q[5])
    x18 = sin(q[4])
    x19 = x12 * x9
    x20 = x5 * x8
    x21 = x11 * x20 - x19
    x22 = x18 * x21
    x23 = cos(q[4])
    x24 = x1 * x7
    x25 = x14 * x2 - x24 * x5
    x26 = x23 * x25
    x27 = x22 + x26
    x28 = 0.069 * x5
    x29 = 0.069 * x9
    x30 = 0.36435 * x1
    x31 = x11 * x12
    x32 = x3 * x9
    x33 = -x11 * x19 + x20
    x34 = x33 * x7
    x35 = -x32 - x34
    x36 = x10 * x11 + x13
    x37 = x18 * x36
    x38 = x2 * x33 - x24 * x9
    x39 = x23 * x38
    x40 = x37 + x39
    x41 = x11 * x2
    x42 = x1 * x12
    x43 = x42 * x7
    x44 = x41 + x43
    x45 = x1 * x8
    x46 = x18 * x45
    x47 = x11 * x7 - x2 * x42
    x48 = x23 * x47
    x49 = x46 + x48
    #
    pose[0] = x0 * x16 + x17 * x27
    pose[1] = -x18 * x25 + x21 * x23
    pose[2] = x0 * x27 - x16 * x17
    pose[3] = 0.37429 * x15 + 0.01 * x22 + 0.01 * x26 - x28 * x31 + x28 - x29 * x8 + x30 * x5 + 0.37429 * x6
    pose[4] = x0 * x35 + x17 * x40
    pose[5] = -x18 * x38 + x23 * x36
    pose[6] = x0 * x40 - x17 * x35
    pose[7] = x28 * x8 - x29 * x31 + x29 + x30 * x9 + 0.37429 * x32 + 0.37429 * x34 + 0.01 * x37 + 0.01 * x39
    pose[8] = x0 * x44 + x17 * x49
    pose[9] = -x18 * x47 + x23 * x45
    pose[10] = x0 * x49 - x17 * x44
    pose[11] = -0.36435 * x11 - 0.37429 * x41 - 0.069 * x42 - 0.37429 * x43 + 0.01 * x46 + 0.01 * x48 + 0.27035
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


def joint_fk06(q):
    #
    pose = [0] * 16
    #
    x0 = sin(q[6])
    x1 = cos(q[4])
    x2 = cos(q[2])
    x3 = q[0] + pi / 4
    x4 = sin(x3)
    x5 = x2 * x4
    x6 = sin(q[1])
    x7 = sin(q[2])
    x8 = cos(x3)
    x9 = x7 * x8
    x10 = -x5 + x6 * x9
    x11 = sin(q[4])
    x12 = sin(q[3])
    x13 = cos(q[1])
    x14 = x12 * x13
    x15 = cos(q[3])
    x16 = x4 * x7
    x17 = x2 * x8
    x18 = -x16 - x17 * x6
    x19 = -x14 * x8 + x15 * x18
    x20 = x1 * x10 - x11 * x19
    x21 = cos(q[6])
    x22 = sin(q[5])
    x23 = x13 * x15
    x24 = x23 * x8
    x25 = x12 * x18
    x26 = -x24 - x25
    x27 = cos(q[5])
    x28 = x10 * x11
    x29 = x1 * x19
    x30 = x28 + x29
    x31 = x22 * x26 + x27 * x30
    x32 = x26 * x27
    x33 = x22 * x30
    x34 = 0.069 * x8
    x35 = 0.069 * x4
    x36 = 0.36435 * x13
    x37 = x2 * x6
    x38 = x16 * x6 + x17
    x39 = -x5 * x6 + x9
    x40 = -x14 * x4 + x15 * x39
    x41 = x1 * x38 - x11 * x40
    x42 = x23 * x4
    x43 = x12 * x39
    x44 = -x42 - x43
    x45 = x11 * x38
    x46 = x1 * x40
    x47 = x45 + x46
    x48 = x22 * x44 + x27 * x47
    x49 = x27 * x44
    x50 = x22 * x47
    x51 = x13 * x7
    x52 = x13 * x2
    x53 = x12 * x6 - x15 * x52
    x54 = x1 * x51 - x11 * x53
    x55 = x15 * x6
    x56 = x12 * x52
    x57 = x55 + x56
    x58 = x11 * x51
    x59 = x1 * x53
    x60 = x58 + x59
    x61 = x22 * x57 + x27 * x60
    x62 = x27 * x57
    x63 = x22 * x60
    #
    pose[0] = x0 * x20 + x21 * x31
    pose[1] = -x0 * x31 + x20 * x21
    pose[2] = -x32 + x33
    pose[
        3] = 0.37429 * x24 + 0.37429 * x25 + 0.01 * x28 + 0.01 * x29 - 0.229525 * x32 + 0.229525 * x33 - x34 * x37 + x34 - x35 * x7 + x36 * x8
    pose[4] = x0 * x41 + x21 * x48
    pose[5] = -x0 * x48 + x21 * x41
    pose[6] = -x49 + x50
    pose[
        7] = x34 * x7 - x35 * x37 + x35 + x36 * x4 + 0.37429 * x42 + 0.37429 * x43 + 0.01 * x45 + 0.01 * x46 - 0.229525 * x49 + 0.229525 * x50
    pose[8] = x0 * x54 + x21 * x61
    pose[9] = -x0 * x61 + x21 * x54
    pose[10] = -x62 + x63
    pose[
        11] = -0.069 * x52 - 0.37429 * x55 - 0.37429 * x56 + 0.01 * x58 + 0.01 * x59 - 0.36435 * x6 - 0.229525 * x62 + 0.229525 * x63 + 0.27035
    pose[12] = 0
    pose[13] = 0
    pose[14] = 0
    pose[15] = 1
    #
    pose = offset_and_reshape(pose, 0.0640273121782, 0.259027312178, 0.129626)
    return pose


FK = {0: joint_fk00, 1: joint_fk01, 2: joint_fk02, 3: joint_fk03, 4: joint_fk04, 5: joint_fk05, 6: joint_fk06, }


def jacobian00(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    #
    jacobian[0] = -0.069 * sin(x0)
    jacobian[1] = 0
    jacobian[2] = 0
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = 0.069 * cos(x0)
    jacobian[8] = 0
    jacobian[9] = 0
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = 0
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = 0
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = 0
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = 0
    jacobian[30] = 0
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = 0
    jacobian[38] = 0
    jacobian[39] = 0
    jacobian[40] = 0
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian01(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = cos(x0)
    #
    jacobian[0] = -0.069 * x1
    jacobian[1] = 0
    jacobian[2] = 0
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = 0.069 * x2
    jacobian[8] = 0
    jacobian[9] = 0
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = 0
    jacobian[16] = 0
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = 0
    jacobian[24] = 0
    jacobian[25] = 0
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = x2
    jacobian[30] = 0
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = 0
    jacobian[38] = 0
    jacobian[39] = 0
    jacobian[40] = 0
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian02(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = 0.069 * x1
    x3 = sin(q[2])
    x4 = cos(x0)
    x5 = 0.069 * x4
    x6 = x3 * x5
    x7 = cos(q[1])
    x8 = x1 * x7
    x9 = 0.36435 * x8
    x10 = sin(q[1])
    x11 = cos(q[2])
    x12 = 0.069 * x10 * x11
    x13 = x1 * x12
    x14 = -0.36435 * x10 - 0.069 * x11 * x7
    x15 = -x13 + x6 + x9
    x16 = x4 * x7
    x17 = -x12 * x4 + 0.36435 * x16 - x2 * x3
    #
    jacobian[0] = x13 - x2 - x6 - x9
    jacobian[1] = x14 * x4
    jacobian[2] = x10 * x15 + x14 * x8
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = x17 + x5
    jacobian[8] = x1 * x14
    jacobian[9] = -x10 * x17 - x14 * x16
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = -x1 * x15 - x17 * x4
    jacobian[16] = x15 * x16 - x17 * x8
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = x16
    jacobian[24] = 0
    jacobian[25] = 0
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = x4
    jacobian[30] = x8
    jacobian[31] = 0
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = -x10
    jacobian[38] = 0
    jacobian[39] = 0
    jacobian[40] = 0
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian03(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = 0.069 * x1
    x3 = sin(q[2])
    x4 = cos(x0)
    x5 = 0.069 * x4
    x6 = x3 * x5
    x7 = cos(q[1])
    x8 = x1 * x7
    x9 = 0.36435 * x8
    x10 = sin(q[1])
    x11 = cos(q[2])
    x12 = 0.069 * x10 * x11
    x13 = x1 * x12
    x14 = -0.36435 * x10 - 0.069 * x11 * x7
    x15 = -x13 + x6 + x9
    x16 = x4 * x7
    x17 = -x12 * x4 + 0.36435 * x16 - x2 * x3
    x18 = x10 * x3
    #
    jacobian[0] = x13 - x2 - x6 - x9
    jacobian[1] = x14 * x4
    jacobian[2] = x10 * x15 + x14 * x8
    jacobian[3] = 0
    jacobian[4] = 0
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = x17 + x5
    jacobian[8] = x1 * x14
    jacobian[9] = -x10 * x17 - x14 * x16
    jacobian[10] = 0
    jacobian[11] = 0
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = -x1 * x15 - x17 * x4
    jacobian[16] = x15 * x16 - x17 * x8
    jacobian[17] = 0
    jacobian[18] = 0
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = x16
    jacobian[24] = -x1 * x11 + x18 * x4
    jacobian[25] = 0
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = x4
    jacobian[30] = x8
    jacobian[31] = x1 * x18 + x11 * x4
    jacobian[32] = 0
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = -x10
    jacobian[38] = x3 * x7
    jacobian[39] = 0
    jacobian[40] = 0
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian04(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = 0.069 * x1
    x3 = sin(q[2])
    x4 = cos(x0)
    x5 = 0.069 * x4
    x6 = x3 * x5
    x7 = cos(q[1])
    x8 = x1 * x7
    x9 = 0.36435 * x8
    x10 = sin(q[1])
    x11 = cos(q[2])
    x12 = x10 * x11
    x13 = x12 * x2
    x14 = cos(q[3])
    x15 = x14 * x8
    x16 = 0.37429 * x15
    x17 = 0.01 * sin(q[4])
    x18 = x11 * x4
    x19 = x1 * x3
    x20 = x10 * x19 + x18
    x21 = x17 * x20
    x22 = sin(q[3])
    x23 = x3 * x4
    x24 = x1 * x11
    x25 = -x10 * x24 + x23
    x26 = x22 * x25
    x27 = 0.37429 * x26
    x28 = 0.01 * cos(q[4])
    x29 = x28 * (x14 * x25 - x22 * x8)
    x30 = x11 * x7
    x31 = x10 * x14
    x32 = x3 * x7
    x33 = x22 * x30
    x34 = x17 * x32 + x28 * (x10 * x22 - x14 * x30) - 0.37429 * x31 - 0.37429 * x33
    x35 = -0.36435 * x10 - 0.069 * x30 + x34
    x36 = x16 + x21 + x27 + x29
    x37 = -x13 + x36 + x6 + x9
    x38 = x15 + x26
    x39 = -x31 - x33
    x40 = x4 * x7
    x41 = x14 * x40
    x42 = x10 * x23 - x24
    x43 = -x10 * x18 - x19
    x44 = x22 * x43
    x45 = x17 * x42 + x28 * (x14 * x43 - x22 * x40) + 0.37429 * x41 + 0.37429 * x44
    x46 = -x12 * x5 - x2 * x3 + 0.36435 * x40 + x45
    x47 = x41 + x44
    #
    jacobian[0] = x13 - x16 - x2 - x21 - x27 - x29 - x6 - x9
    jacobian[1] = x35 * x4
    jacobian[2] = x10 * x37 + x35 * x8
    jacobian[3] = x20 * x34 - x32 * x36
    jacobian[4] = x34 * x38 - x36 * x39
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = x46 + x5
    jacobian[8] = x1 * x35
    jacobian[9] = -x10 * x46 - x35 * x40
    jacobian[10] = x32 * x45 - x34 * x42
    jacobian[11] = -x34 * x47 + x39 * x45
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = -x1 * x37 - x4 * x46
    jacobian[16] = x37 * x40 - x46 * x8
    jacobian[17] = -x20 * x45 + x36 * x42
    jacobian[18] = x36 * x47 - x38 * x45
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = x40
    jacobian[24] = x42
    jacobian[25] = x47
    jacobian[26] = 0
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = x4
    jacobian[30] = x8
    jacobian[31] = x20
    jacobian[32] = x38
    jacobian[33] = 0
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = -x10
    jacobian[38] = x32
    jacobian[39] = x39
    jacobian[40] = 0
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian05(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = 0.069 * x1
    x3 = sin(q[2])
    x4 = cos(x0)
    x5 = 0.069 * x4
    x6 = x3 * x5
    x7 = cos(q[1])
    x8 = x1 * x7
    x9 = 0.36435 * x8
    x10 = sin(q[1])
    x11 = cos(q[2])
    x12 = x10 * x11
    x13 = x12 * x2
    x14 = cos(q[3])
    x15 = x14 * x8
    x16 = 0.37429 * x15
    x17 = sin(q[4])
    x18 = 0.01 * x17
    x19 = x11 * x4
    x20 = x1 * x3
    x21 = x10 * x20 + x19
    x22 = x18 * x21
    x23 = sin(q[3])
    x24 = x3 * x4
    x25 = x1 * x11
    x26 = -x10 * x25 + x24
    x27 = x23 * x26
    x28 = 0.37429 * x27
    x29 = cos(q[4])
    x30 = 0.01 * x29
    x31 = x14 * x26 - x23 * x8
    x32 = x30 * x31
    x33 = x11 * x7
    x34 = x10 * x14
    x35 = x3 * x7
    x36 = x23 * x33
    x37 = x10 * x23 - x14 * x33
    x38 = x18 * x35 + x30 * x37 - 0.37429 * x34 - 0.37429 * x36
    x39 = -0.36435 * x10 - 0.069 * x33 + x38
    x40 = x16 + x22 + x28 + x32
    x41 = -x13 + x40 + x6 + x9
    x42 = x15 + x27
    x43 = -x34 - x36
    x44 = x4 * x7
    x45 = x14 * x44
    x46 = x10 * x24 - x25
    x47 = -x10 * x19 - x20
    x48 = x23 * x47
    x49 = x14 * x47 - x23 * x44
    x50 = x18 * x46 + x30 * x49 + 0.37429 * x45 + 0.37429 * x48
    x51 = -x12 * x5 - x2 * x3 + 0.36435 * x44 + x50
    x52 = x45 + x48
    #
    jacobian[0] = x13 - x16 - x2 - x22 - x28 - x32 - x6 - x9
    jacobian[1] = x39 * x4
    jacobian[2] = x10 * x41 + x39 * x8
    jacobian[3] = x21 * x38 - x35 * x40
    jacobian[4] = x38 * x42 - x40 * x43
    jacobian[5] = 0
    jacobian[6] = 0
    jacobian[7] = x5 + x51
    jacobian[8] = x1 * x39
    jacobian[9] = -x10 * x51 - x39 * x44
    jacobian[10] = x35 * x50 - x38 * x46
    jacobian[11] = -x38 * x52 + x43 * x50
    jacobian[12] = 0
    jacobian[13] = 0
    jacobian[14] = 0
    jacobian[15] = -x1 * x41 - x4 * x51
    jacobian[16] = x41 * x44 - x51 * x8
    jacobian[17] = -x21 * x50 + x40 * x46
    jacobian[18] = x40 * x52 - x42 * x50
    jacobian[19] = 0
    jacobian[20] = 0
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = x44
    jacobian[24] = x46
    jacobian[25] = x52
    jacobian[26] = -x17 * x49 + x29 * x46
    jacobian[27] = 0
    jacobian[28] = 0
    jacobian[29] = x4
    jacobian[30] = x8
    jacobian[31] = x21
    jacobian[32] = x42
    jacobian[33] = -x17 * x31 + x21 * x29
    jacobian[34] = 0
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = -x10
    jacobian[38] = x35
    jacobian[39] = x43
    jacobian[40] = -x17 * x37 + x29 * x35
    jacobian[41] = 0
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


def jacobian06(q):
    #
    jacobian = [0] * 42
    #
    x0 = q[0] + pi / 4
    x1 = sin(x0)
    x2 = 0.069 * x1
    x3 = sin(q[2])
    x4 = cos(x0)
    x5 = 0.069 * x4
    x6 = x3 * x5
    x7 = cos(q[1])
    x8 = x1 * x7
    x9 = 0.36435 * x8
    x10 = sin(q[1])
    x11 = cos(q[2])
    x12 = x10 * x11
    x13 = x12 * x2
    x14 = cos(q[3])
    x15 = x14 * x8
    x16 = 0.37429 * x15
    x17 = sin(q[4])
    x18 = x11 * x4
    x19 = x1 * x3
    x20 = x10 * x19 + x18
    x21 = x17 * x20
    x22 = 0.01 * x21
    x23 = sin(q[3])
    x24 = x3 * x4
    x25 = x1 * x11
    x26 = -x10 * x25 + x24
    x27 = x23 * x26
    x28 = 0.37429 * x27
    x29 = cos(q[4])
    x30 = x14 * x26 - x23 * x8
    x31 = x29 * x30
    x32 = 0.01 * x31
    x33 = cos(q[5])
    x34 = x33 * (-x15 - x27)
    x35 = 0.229525 * x34
    x36 = sin(q[5])
    x37 = x36 * (x21 + x31)
    x38 = 0.229525 * x37
    x39 = x11 * x7
    x40 = x10 * x14
    x41 = x3 * x7
    x42 = x17 * x41
    x43 = x23 * x39
    x44 = x10 * x23 - x14 * x39
    x45 = x29 * x44
    x46 = x33 * (x40 + x43)
    x47 = x36 * (x42 + x45)
    x48 = -0.229525 * x46 + 0.229525 * x47
    x49 = -0.37429 * x40 + 0.01 * x42 - 0.37429 * x43 + 0.01 * x45 + x48
    x50 = -0.36435 * x10 - 0.069 * x39 + x49
    x51 = -x35 + x38
    x52 = x16 + x22 + x28 + x32 + x51
    x53 = -x13 + x52 + x6 + x9
    x54 = x15 + x27
    x55 = -x40 - x43
    x56 = -x17 * x30 + x20 * x29
    x57 = -x17 * x44 + x29 * x41
    x58 = -x34 + x37
    x59 = -x46 + x47
    x60 = x4 * x7
    x61 = x14 * x60
    x62 = x10 * x24 - x25
    x63 = x17 * x62
    x64 = -x10 * x18 - x19
    x65 = x23 * x64
    x66 = x14 * x64 - x23 * x60
    x67 = x29 * x66
    x68 = x33 * (-x61 - x65)
    x69 = x36 * (x63 + x67)
    x70 = -0.229525 * x68 + 0.229525 * x69
    x71 = 0.37429 * x61 + 0.01 * x63 + 0.37429 * x65 + 0.01 * x67 + x70
    x72 = -x12 * x5 - x2 * x3 + 0.36435 * x60 + x71
    x73 = x61 + x65
    x74 = -x17 * x66 + x29 * x62
    x75 = -x68 + x69
    #
    jacobian[0] = x13 - x16 - x2 - x22 - x28 - x32 + x35 - x38 - x6 - x9
    jacobian[1] = x4 * x50
    jacobian[2] = x10 * x53 + x50 * x8
    jacobian[3] = x20 * x49 - x41 * x52
    jacobian[4] = x49 * x54 - x52 * x55
    jacobian[5] = x48 * x56 - x51 * x57
    jacobian[6] = x48 * x58 - x51 * x59
    jacobian[7] = x5 + x72
    jacobian[8] = x1 * x50
    jacobian[9] = -x10 * x72 - x50 * x60
    jacobian[10] = x41 * x71 - x49 * x62
    jacobian[11] = -x49 * x73 + x55 * x71
    jacobian[12] = -x48 * x74 + x57 * x70
    jacobian[13] = -x48 * x75 + x59 * x70
    jacobian[14] = 0
    jacobian[15] = -x1 * x53 - x4 * x72
    jacobian[16] = x53 * x60 - x72 * x8
    jacobian[17] = -x20 * x71 + x52 * x62
    jacobian[18] = x52 * x73 - x54 * x71
    jacobian[19] = x51 * x74 - x56 * x70
    jacobian[20] = x51 * x75 - x58 * x70
    jacobian[21] = 0
    jacobian[22] = -x1
    jacobian[23] = x60
    jacobian[24] = x62
    jacobian[25] = x73
    jacobian[26] = x74
    jacobian[27] = x75
    jacobian[28] = 0
    jacobian[29] = x4
    jacobian[30] = x8
    jacobian[31] = x20
    jacobian[32] = x54
    jacobian[33] = x56
    jacobian[34] = x58
    jacobian[35] = 1
    jacobian[36] = 0
    jacobian[37] = -x10
    jacobian[38] = x41
    jacobian[39] = x55
    jacobian[40] = x57
    jacobian[41] = x59
    #
    jacobian = np.array(jacobian).reshape(6, 7)
    return jacobian


J = {0: jacobian00, 1: jacobian01, 2: jacobian02, 3: jacobian03, 4: jacobian04, 5: jacobian05, 6: jacobian06, }
