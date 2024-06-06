import numpy as np
from scipy import stats
from PIL import Image
from math import floor, sqrt
import sys
import os
from PIL import Image, ImageDraw
from cryptography.fernet import Fernet
from colorama import Fore, Style

def chi_squared_test(channel):
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


def encrypt(path_to_image, text, key, balance, out_filename="out.png"):
    """
    Encrypt in image
    :param path_to_image: path
    :param text: text from data file
    :param key: generated key
    :param balance: 1 to 4
    :return:
    """
    img = dict()
    size = dict()
    coord = dict()

    img["image"] = Image.open(path_to_image)
    img["draw"] = ImageDraw.Draw(img["image"])
    img["pix"] = img["image"].load()

    size["width"] = img["image"].size[0]
    size["height"] = img["image"].size[1]

    text = des_encrypt(text, key)
    binary_text = text_to_binary(text)
    list_two = split_count(''.join(binary_text), balance)

    coord["x"] = 0
    coord["y"] = 0
    count = 0

    for i in list_two:
        red, green, blue = img["pix"][coord["x"], coord["y"]]

        (red, green, blue) = balance_channel([red, green, blue], i)

        img["draw"].point((coord["x"], coord["y"]), (red, green, blue))

        if coord["x"] < (size["width"] - 1):
            coord["x"] += 1

        elif coord["y"] < (size["height"] - 1):
            coord["y"] += 1
            coord["x"] = 0

        else:
            error("Message too long for this image.", True)

        count += 1

    img["image"].save(out_filename, "PNG")

    file = open("key.dat", "w")
    file.write(str(balance) + '$' + str(count) + '$' + key)
    file.close()




def decrypt(path_to_image, key):
    """

    :param path_to_image:
    :param key:
    :return:
    """
    balance = int(key.split('$')[0])
    count = int(key.split('$')[1])
    end_key = key.split('$')[2]

    img = dict()
    coord = dict()

    img["image"] = Image.open(path_to_image)
    img["width"] = img["image"].size[0]
    img["height"] = img["image"].size[1]
    img["pix"] = img["image"].load()

    coord["x"] = 0
    coord["y"] = 0
    code = ''

    i = 0
    while i < count:
        pixels = img["pix"][coord["x"], coord["y"]]

        pixel = str(bin(max(pixels)))

        if balance == 4:
            code += pixel[-4] + pixel[-3] + pixel[-2] + pixel[-1]

        elif balance == 3:
            code += pixel[-3] + pixel[-2] + pixel[-1]

        elif balance == 2:
            code += pixel[-2] + pixel[-1]

        else:
            code += pixel[-1]

        if coord["x"] < (img["width"] - 1):
            coord["x"] += 1
        else:
            coord["y"] += 1
            coord["x"] = 0

        i += 1

    outed = binary_to_text(split_count(code, 8))
    str1=des_decrypt(''.join(outed), end_key)
    # file = open("out.txt", "w")
    # file.write(str1)
    # file.close()
    with open("out.txt", "w", encoding='utf-8') as file:
        file.write(str1)
    
    
    success("Data saved in out.txt")


def des_encrypt(text, key):
    """
    DES Encrypting
    :param text:
    :param key:
    :return: encrypt data
    """
    cipher = Fernet(key.encode('utf-8'))
    result = cipher.encrypt(text.encode('utf-8'))

    return result.decode('utf-8')


def des_decrypt(text, key):
    """
    DES Decrypting
    :param text:
    :param key:
    :return:
    """
    cipher = Fernet(key.encode('utf-8'))
    result = cipher.decrypt(text.encode('utf-8'))

    return result.decode('utf-8')


def split_count(text, count):
    """
    Splitting every count
    :param text:
    :param count:
    :return:
    """
    result = list()
    txt = ''
    var = 0

    for i in text:
        if var == count:
            result.append(txt)
            txt = ''
            var = 0

        txt += i
        var += 1

    result.append(txt)

    return result


def last_replace(main_string, last_symbols):
    """
    Replace string with substring, starting from the end
    :param main_string: пиздец
    :param last_symbols: бля
    :return: пизбля
    """
    return str(main_string)[:-len(last_symbols)] + last_symbols

def text_to_binary(event):
    """
    Text convert to binary code
    :param event: text
    :return: binary code(str)
    """
    return ['0' * (8 - len(format(ord(elem), 'b'))) + format(ord(elem), 'b') for elem in event]


def binary_to_text(event):
    """
    Binary code convert to text
    :param event: binary code(str)
    :return: text
    """
    return [chr(int(str(elem), 2)) for elem in event]


def isset(array, key):
    """
    Сheck for the existence of a key in an list/dict
    :param array: dict or list
    :param key: key in dict/list
    :return: boolean
    """
    try:
        if type(array) is list:
            array[key]

        elif type(array) is dict:
            return key in array.keys()

        return True
    except:
        return False


def error(text, quit=False):
    """
    Print a customized error
    :param text: error text
    """
    print(Style.BRIGHT + Fore.YELLOW + "     " + text + Style.RESET_ALL)

    if quit:
        sys.exit()


def using(text, quit=False):
    """
    Print a customized using message
    :param text: using message
    """
    print(Style.BRIGHT + Fore.WHITE + "     " + text + Style.RESET_ALL)

    if quit:
        sys.exit()


def success(text):
    """
    Print a customized successful message
    :param text: success message
    """
    print(Style.BRIGHT + Fore.GREEN + "     " + text + Style.RESET_ALL)


def find_max_index(array):
    """
    Find max number and return its index
    :param array: input array
    :return: max element index
    """
    max_num = array[0]
    index = 0

    for i, val in enumerate(array):
        if val > max_num:
            max_num = val
            index = i

    return index


def balance_channel(colors, pix):
    """
    Find an optimal channel to write data
    :param colors: red, green and blue channels
    :param pix: data to write
    :return: modified colors array
    """
    max_color = find_max_index(colors)
    colors[max_color] = int(last_replace(bin(colors[max_color]), pix), 2)

    while True:
        max_sec = find_max_index(colors)
        if max_sec != max_color:
            colors[max_sec] = colors[max_color] - 1
        else:
            break

    return colors

def encode(fileName, out_filename, text, balance=1):
    # balance 1..4
    encrypt(fileName, text.strip(), Fernet.generate_key().decode(), balance, out_filename)

        
if __name__ == "__main__":
    img_fileName = "d://ml/a1.png"
    out_filename="d://ml/out.png" 
    text = '1111111111111111111111111!'
    with open("data.txt", "r", encoding='utf-8') as file:
            text = file.read()
    
    # visual_attack(Image.open(img_fileName))
    # # chi_squared_attack(Image.open("a1.png"))
    # z=spa_test(Image.open(img_fileName))
    # print(z)
    # z=rs_test(Image.open(img_fileName))
    # print(z)
    
    encode(img_fileName, out_filename, text,4)
    key=""
    with open("key.dat", "r", encoding='utf-8') as file:
        key = file.read()
    s=decrypt(out_filename, key)
    print(s)
    
    # visual_attack(Image.open(out_filename))
    z=spa_test(Image.open(out_filename))
    print(z)
    z=rs_test(Image.open(out_filename))
    print(z)
    
    img_fileName = "d://ml/a1.png" 
    # visual_attack(Image.open(img_fileName))
    z=spa_test(Image.open(img_fileName))
    print(z)
    z=rs_test(Image.open(img_fileName))
    print(z)