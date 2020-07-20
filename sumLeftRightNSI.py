import numpy as np
import sys


def SumLeftRightEqualWeights(data):
  left_plus_right = np.empty((data.shape[0], data.shape[1]-6))
  left_plus_right[:, 0] = data[:, 0] + data[:, 1]  # ee
  left_plus_right[:, 1] = data[:, 2] + data[:, 3]  # mm
  left_plus_right[:, 2] = data[:, 4] + data[:, 5]  # tt
  left_plus_right[:, 3] = data[:, 6] + data[:, 7]  # em
  left_plus_right[:, 4] = data[:, 8] + data[:, 9]  # et
  left_plus_right[:, 5] = data[:, 10] + data[:, 11]  # mt
  return left_plus_right


def SumLeftRight(data):
  left_plus_right = np.empty((data.shape[0], data.shape[1]-6))
  left_plus_right[:, 0] = data[:, 0]
  left_plus_right[:, 1] = data[:, 1]
  left_plus_right[:, 2] = data[:, 2] + data[:, 3]  # ee
  left_plus_right[:, 3] = data[:, 4] + data[:, 5]  # mm
  left_plus_right[:, 4] = data[:, 6] + data[:, 7]  # tt
  left_plus_right[:, 5] = data[:, 8] + data[:, 9]  # em
  left_plus_right[:, 6] = data[:, 10] + data[:, 11]  # et
  left_plus_right[:, 7] = data[:, 12] + data[:, 13]  # mt
  return left_plus_right

def SumLeftRightDiag(data):
  left_plus_right = np.empty((data.shape[0], data.shape[1] - 3))
  left_plus_right[:, 0] = data[:, 0]
  left_plus_right[:, 1] = data[:, 1]
  left_plus_right[:, 2] = data[:, 2] + data[:, 5]  # ee
  left_plus_right[:, 3] = data[:, 3] + data[:, 6]  # mm
  left_plus_right[:, 4] = data[:, 4] + data[:, 7]  # tt
  left_plus_right[:, 5] = data[:, 8]  # u_ee
  left_plus_right[:, 6] = data[:, 9]  # u_mm
  left_plus_right[:, 7] = data[:, 10]  # u_tt
  left_plus_right[:, 8] = data[:, 11]  # d_ee
  left_plus_right[:, 9] = data[:, 12]  # d_mm
  left_plus_right[:, 10] = data[:, 13]  # d_tt
  return left_plus_right


def SumUpDownLeftRightDiag(data):
  left_plus_right = np.empty((data.shape[0], data.shape[1] - 6))
  left_plus_right[:, 0] = data[:, 0]
  left_plus_right[:, 1] = data[:, 1]
  left_plus_right[:, 2] = data[:, 2] + data[:, 5]  # ee
  left_plus_right[:, 3] = data[:, 3] + data[:, 6]  # mm
  left_plus_right[:, 4] = data[:, 4] + data[:, 7]  # tt
  left_plus_right[:, 5] = data[:, 8] + data[:, 11]  # u_ee
  left_plus_right[:, 6] = data[:, 9] + data[:, 12]  # u_mm
  left_plus_right[:, 7] = data[:, 10] + data[:, 13]  # u_tt


def SumUpDown(data):
  out = np.empty((data.shape[0], data.shape[1] - 6))
  out[:, 0] = data[:, 0]
  out[:, 1] = data[:, 1]
  out[:, 2] = data[:, 2] + data[:, 8]  # ee
  out[:, 3] = data[:, 3] + data[:, 9]  # mm
  out[:, 4] = data[:, 4] + data[:, 10]  # tt
  out[:, 5] = data[:, 5] + data[:, 11]  # u_ee
  out[:, 6] = data[:, 6] + data[:, 12]  # u_mm
  out[:, 7] = data[:, 7] + data[:, 13]  # u_tt
  return out

def SumUpDownNoTau(data):
  out = np.empty((data.shape[0], data.shape[1] - 5))
  out[:, 0] = data[:, 0]
  out[:, 1] = data[:, 1]
  out[:, 2] = data[:, 2] + data[:, 7]  # ee
  out[:, 3] = data[:, 3] + data[:, 8]  # mm
  out[:, 4] = data[:, 4] + data[:, 9]  # em
  out[:, 5] = data[:, 5] + data[:, 10]  # et
  out[:, 6] = data[:, 6] + data[:, 11]  # mt
  return out

def SumUpDown18to12(data):
  out = np.empty((data.shape[0], data.shape[1] - 6))
  out[:, 0] = data[:, 0]
  out[:, 1] = data[:, 1]
  out[:, 2] = data[:, 2]
  out[:, 3] = data[:, 3]
  out[:, 4] = data[:, 4]
  out[:, 5] = data[:, 5]
  out[:, 6] = data[:, 6]
  out[:, 7] = data[:, 7]
  out[:, 8] = data[:, 8] + data[:, 14]  # ee
  out[:, 9] = data[:, 9] + data[:, 15]  # mm
  out[:, 10] = data[:, 10] + data[:, 16]  # tt
  out[:, 11] = data[:, 11] + data[:, 17]  # u_ee
  out[:, 12] = data[:, 12] + data[:, 18]  # u_mm
  out[:, 13] = data[:, 13] + data[:, 19]  # u_tt
  return out

# Sum left and right electron NSI, use L inverse to transform pheno into physical NSI, then sum up and down.
def SumTransformSum(data):
  n = 78
  z = 54
  lmatr = np.array([
    [1, 0, 0],
    [0, 1, (2 * n + z) / (2 * z + n)],
    [1, 3, 3]
  ])

  linv = np.inv(lmatr)

  out = np.empty((data.shape[0], 12))
  out[:, 0] = data[:, 0]
  out[:, 1] = data[:, 1]
  out[:, 2] = data[:, 2] + data[:, 5]  # ee
  out[:, 3] = data[:, 3] + data[:, 6]  # mm
  out[:, 4] = data[:, 4] + data[:, 7]  # tt
  out[:, 5] = data[:, 8] + data[:, 11]  # em
  out[:, 6] = data[:, 9] + data[:, 12]  # et
  out[:, 7] = data[:, 10] + data[:, 13]  # mt

  out[:,8] = np.dot(linv[1], np.array([out[:,2], data[:,14], data[:,20]]))


def main(infile, outfile):
  data = np.genfromtxt(infile)
  
  transformed = SumUpDown(data)

  np.savetxt(outfile, transformed)


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
