import numpy as np
from scipy import stats
from PIL import Image
from math import sqrt

def chi_squared_test(channel):
    """Main function for the attack

    Using chi-squared implementation
    from scipy.

    :param channel: Channel for analyzing 

    """
    hist = calc_colors(channel)
    expected_freq, observed_freq = calc_freq(hist)
    chis, probs = cs(observed_freq, expected_freq) 
    return chis, probs

def cs(n, y):
    n = np.asarray(n)
    y= np.asarray(y)
    z = np.sum(n)/np.sum(y) * y
    return stats.chisquare(n, z)

def calc_colors(channel):
    """Prepare color histogram for further calculations"""
    hist = channel.histogram()
    hist = list(map(lambda x: 1 if x == 0 else x, hist)) # to avoid dividing by zero 
    return hist

def calc_freq(histogram):
    """Calculating expacted and observed freqs"""
    expected = []
    observed = []
    for k in range(0, len(histogram) // 2):
        expected.append((histogram[2 * k] + histogram[2 * k + 1]) / 2)
        observed.append(histogram[2 * k])

    return expected, observed


def visual_attack(img, join=False, bitnum=1):
    """Implementing a visual attack

    Visual attack can be of two kinds.
    In the first case, three images of channels with LSB 
    are created, in the second, these three images are 
    combined into one. Images are opened by means of 
    the operating system.

    :param img: Image for attack
    :param join: Is it necessary to divide the image into channels
    :param bitnum: How many LSBs need to take

    """
    # logging.info('Visualising lsb for '+ img.filename +' ...🌀')
    height, width = img.size

    if join == False:
        channels = img.split()
        colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255)] # red, green, blue
        suffixes = ['red', 'green', 'blue']

        for k in range(3):
            channel = channels[k].load()
            img_ch = Image.new("RGB", (height, width), color=colors[k])

            for i in range(height):
                for j in range(width):
                    bin_channel = bin(channel[i, j])
                    bin_channel = bin_channel[:2] + '0'*(10-len(bin_channel)) + bin_channel[2:]
                    bit = int(bin_channel[-bitnum]) # takes LSB
                    if bit == 1:
                        if k == 0:
                            img_ch.putpixel((i, j), 255) # black
                        else:
                            img_ch.putpixel((i, j), 0) # white
            name = "ggg.png"
            # name = suffixes[k] + "-" + img.filename.split(".")[0] + ".bmp"
            img_ch.save(name)
            # logging.info("Openning " + suffixes[k] + " component...🌀")
            img_ch.show()
    else:
        img_ch = Image.new("RGB", (height, width), color=(0, 0, 0))
        for i in range(height):
            for j in range(width):
                pixel = img.getpixel((i, j))
                if len(pixel) == 4: # if RGBA
                    pixel = pixel[:-1]
                new_pixel = [0, 0, 0]
                for k in range(3):
                    if int(bin(pixel[k])[-1]) == 1:
                        new_pixel[k] = 255
                    else:
                        new_pixel[k] = 0

                img_ch.putpixel((i, j), tuple(new_pixel))
        img_ch.save("LSB-" + img.filename.split(".")[0] + ".bmp")
        # logging.info("Openning LSB image...🌀")
        img_ch.show()

def chi_squared_attack( img, eps=1e-5):
    """Implementing a chi-squared attack

    Westfeld and Pfitzmann attack using chi-square test.
    The paper describing this method can be found here:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.94.5975&rep=rep1&type=pdf 
    The test is applied to each line, then the original image 
    is colored in accordance with the result. 
    A new image with a test result is saved in the current directory.


    :param img: Image for attack
    :param eps: Error value for probability  (default is 1e-5)
    
    """
    # logging.info('Calculating chi_squared for '+ img.filename +' ...🌀')
    channels = img.split()
    width, height = img.size

    img_to_blend = Image.new(img.mode, (width, height), color=(0, 0, 0)) # image with results 

    for i in range(height):
        prob = 0
        for ch in channels:
            data = ch.crop((0, i, width, i+1)) # crop for new line 
            prob += chi_squared_test(data)[1]
            # print(prob)
        prob /= 3
        if 0.5 - eps < prob < 0.5 + eps: 
            for j in range(width):
                img_to_blend.putpixel((j, i), (209, 167, 27)) # yellow
        elif prob < 0.5 - eps:
            for j in range(width):
                img_to_blend.putpixel((j, i), (112, 209, 27)) # green
        elif prob > 0.5 + eps:
            for j in range(width):
                img_to_blend.putpixel((j, i), (255, 0, 0)) # red

    result = Image.blend(img, img_to_blend, 0.4)
    result.save("chi-alina.png")
    result.show()
    
def spa_test(img):
    """Using the Sample pairs method
    
    :param img: Image name
    """
    height, width = img.size

    if width % 2  == 1:
        width -= 1
    if height % 2  == 1:
        height -= 1


    average = 0.0

    r, g, b = img.split()[:3]

    average = analyze(r.load(), height, width)
    average += analyze(g.load(), height, width, channel='g')
    average += analyze(b.load(), height, width, channel='b')

    average = average / 3.0
    average = abs(average)
    if average > 1:
        return 1
    else:
        return average
    
    
def analyze(pix, h, w, channel='r'):
    """Container analysis for steganography

    :param pix: Image pixels
    :param h: Image height
    :param w: Image width
    :param channel: One of three possible RGB channels. Set 'r' for red, 'g' for green, 'b' for blue
    """
    P = 0 
    X = 0 
    Y = 0 
    Z = 0
    W = 0

    for i in range(0, h - 1):
        for j in range(0, w - 1, 2):
            u = pix[i, j]
            v = pix[i + 1, j]

            if (u >> 1 == v >> 1) and ((v & 0x1) != (u & 0x1)):
                W += 1

            if u == v:
                Z += 1

            # if lsb(v) = 0 & u < v OR lsb(v) = 1 & u > v
            if (v == (v >> 1) << 1) and (u < v) or (v != (v >> 1) << 1) and (u > v):
                X += 1

            # vice versa
            if (v == (v >> 1) << 1) and (u > v) or (v != (v >> 1) << 1) and (u < v):
                Y += 1
            
            P += 1 

    for i in range(0, h - 1, 2):
        for j in range(0, w - 1):
            u = pix[i, j]
            v = pix[i, j + 1]

            if (u >> 1 == v >> 1) and ((v & 0x1) != (u & 0x1)):
                W += 1

            if u == v:
                Z += 1

            # if lsb(v) = 0 & u < v OR lsb(v) = 1 & u > v
            if (v == (v >> 1) << 1) and (u < v) or (v != (v >> 1) << 1) and (u > v):
                X += 1

            # vice versa
            if (v == (v >> 1) << 1) and (u > v) or (v != (v >> 1) << 1) and (u < v):
                Y += 1
            
            P += 1 
    
    a = 0.5 * (W + Z)
    b = 2 * X - P
    c = Y - X

    if a == 0:
        x = c / b
    
    discriminant = b ** 2 - (4 * a * c)
    if discriminant >= 0:
        rootpos = ((-1 * b) + sqrt(discriminant)) / (2 * a)
        rootneg = ((-1 * b) - sqrt(discriminant)) / (2 * a)

        if abs(rootpos) <= abs(rootneg):
            x = rootpos
        else:
            x = rootneg
    else:
        x = c / b

    if x == 0:
        x = c / b

    return x

from PIL import Image
from math import floor, sqrt

def rs_test(img, bw=2, bh=2, mask=[1, 0, 0, 1]):
    height, width = img.size

    invert_mask = list(map(lambda x: -x, mask))

    blocks_in_row = floor(width / bw)
    blocks_in_col = floor(height / bh)
    r, g, b = img.split()[:3]
    r = r.load()
    g = g.load()
    b = b.load() 

    group_couters = [
		{'R': 0, 'S': 0, 'U': 0, 'mR': 0, 'mS': 0, 'mU': 0, 'iR': 0, 'iS': 0, 'iU': 0, 'imR': 0, 'imS': 0, 'imU': 0},
		{'R': 0, 'S': 0, 'U': 0, 'mR': 0, 'mS': 0, 'mU': 0, 'iR': 0, 'iS': 0, 'iU': 0, 'imR': 0, 'imS': 0, 'imU': 0},
		{'R': 0, 'S': 0, 'U': 0, 'mR': 0, 'mS': 0, 'mU': 0, 'iR': 0, 'iS': 0, 'iU': 0, 'imR': 0, 'imS': 0, 'imU': 0}]


    for y in range(blocks_in_col):
        for x in range(blocks_in_row):
            counter_R = []
            counter_G = []
            counter_B = []
            for v in range(bh): 
                for h in range(bw):
                    counter_R.append(r[y + v, x + h])  # not vice versa?
                    counter_G.append(g[y + v, x + h])
                    counter_B.append(b[y + v, x + h])

            group_couters[0][get_group(counter_R, mask)] += 1
            group_couters[1][get_group(counter_G, mask)] += 1
            group_couters[2][get_group(counter_B, mask)] += 1
            group_couters[0]['m' + get_group(counter_R, invert_mask)] += 1
            group_couters[1]['m' + get_group(counter_G, invert_mask)] += 1
            group_couters[2]['m' + get_group(counter_B, invert_mask)] += 1

            counter_R = lsb_flip(counter_R)
            counter_G = lsb_flip(counter_G)
            counter_B = lsb_flip(counter_B)

            group_couters[0]['i' + get_group(counter_R, mask)] += 1
            group_couters[1]['i' + get_group(counter_G, mask)] += 1
            group_couters[2]['i' + get_group(counter_B, mask)] += 1
            group_couters[0]['im' + get_group(counter_R, invert_mask)] += 1
            group_couters[1]['im' + get_group(counter_G, invert_mask)] += 1
            group_couters[2]['im' + get_group(counter_B, invert_mask)] += 1

    return (solve(group_couters[0]) + solve(group_couters[1]) + solve(group_couters[2])) / 3


def get_group(pix, mask):
    flip_pix = pix[:]

    for i in range(len(mask)):
        if mask[i] == 1:
            flip_pix[i] = flip(pix[i])
        elif mask[i] == -1:
            flip_pix[i] = invert_flip(pix[i])

    d1 = smoothness(pix)
    d2 = smoothness(flip_pix)

    if d1 >  d2: 
        return 'S'
    
    if d1 < d2:
        return 'R'

    return 'U'


def flip(val):
    if val & 1:
        return val - 1

    return val + 1


def invert_flip(val):
    if val & 1:
        return val + 1

    return val - 1


def smoothness(pix):
    s = 0
    for i in range(len(pix) - 1):
        s += abs(pix[i + 1] - pix[i])

    return s


def lsb_flip(pix):
    return list(map(lambda x: x ^ 1, pix))


def solve(groups):
    d0 = groups['R'] - groups['S']
    dm0 = groups['mR'] - groups['mS']
    d1  = groups['iR']  - groups['iS']
    dm1 = groups['imR']  - groups['imS']
    a = 2 * (d1 + d0)
    b = dm0 - dm1 - d1 - d0 * 3
    c = d0 - dm0

    D = b * b - 4 * a * c

    if D < 0:
        return 0 

    b *= -1

    if D == 0:
        return (b / 2 / a) / (b / 2 / a - 0.5)

    D = sqrt(D)

    x1 = (b + D) / 2 / a
    x2 = (b - D) / 2 / a

    if abs(x1) < abs(x2):
        return x1 / (x1 - 0.5)

    return x2 / (x2 - 0.5)

        
if __name__ == "__main__":
    img_fileName = "d://st/a1.png" 

    visual_attack(Image.open(img_fileName))
    # chi_squared_attack(Image.open("a1.png"))
    z=spa_test(Image.open(img_fileName))
    print(z)
    z=rs_test(Image.open(img_fileName))
    print(z)