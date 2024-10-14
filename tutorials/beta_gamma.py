import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma
import scipy
from src.trig_functions import min_max_normalize_array


NUM = 100
PEAK_STEAPN = 1.5
pdf = beta.pdf(x=np.linspace(0, 1, NUM), a=4, b=4, loc=0)
pdf = min_max_normalize_array(pdf, y_range=[0, PEAK_STEAPN])
peak = scipy.signal.find_peaks(pdf)[0][0]

# beta_pdf[peak:] *= np.geomspace(start=1, stop=0.2, num=NUM - peak)
# beta_pdf[peak:] *= np.geomspace(start=1, stop=0.2, num=NUM - peak)

pdf_post_peak = np.exp(np.linspace(start=-0, stop=-10, num=NUM - peak))
pdf_post_peak = min_max_normalize_array(pdf_post_peak, y_range=[0, PEAK_STEAPN])
pdf[peak:] = pdf_post_peak
peak_pdf_val = pdf_post_peak[0]

shift_post_peak_pdf = -beta.pdf(x=np.linspace(0, 1, len(pdf_post_peak)), a=2, b=4, loc=0)
shift_post_peak_pdf = min_max_normalize_array(shift_post_peak_pdf, y_range=[-PEAK_STEAPN * 4, 0])
pdf[peak:] += shift_post_peak_pdf
# beta_pdf = beta_pdf[::-1]

ax0 = plt.plot(pdf, marker='o')
# ax0 = plt.plot(shift_post_peak_pdf, marker='o')

# beta_rvs = beta.rvs(a=2, b=5, loc=0, scale=200, size=25000)
# plt.hist(beta_rvs, bins=100)

# _gamma = gamma.pdf(np.linspace(0, 100, 100), 2, 5, 10)
# ax0 = plt.plot(_gamma)

plt.show()





