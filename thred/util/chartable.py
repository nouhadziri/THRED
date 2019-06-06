
s = '!"#$%&\'()*+,-./01234567890:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
capital_alphabet = "ABCDEFGHIJKLMNOPQRSTUVXYZ"


def _build_halfwidth():
    charmap = {}

    msb, mb, lsb = tuple('\uff01'.encode('utf-8'))
    for ch in s:
        charmap[bytes([msb, mb, lsb]).decode('utf-8')] = ch

        lsb += 1
        if lsb == 0xc0:
            lsb = 0x80
            mb += 1

    assert len(charmap) == len(s)
    return charmap


def _build_enclosed_supplements():

    charmap = {
        bytes([0xf0, 0x9f, 0x84, 0x80]).decode('utf-8'): '0.',
        bytes([0xf0, 0x9f, 0x84, 0x8B]).decode('utf-8'): '0',
        bytes([0xf0, 0x9f, 0x84, 0x8C]).decode('utf-8'): '0',
        bytes([0xf0, 0x9f, 0x84, 0xAA]).decode('utf-8'): '[S]',
        bytes([0xf0, 0x9f, 0x84, 0xAB]).decode('utf-8'): 'C',
        bytes([0xf0, 0x9f, 0x84, 0xAC]).decode('utf-8'): 'R',
        bytes([0xf0, 0x9f, 0x84, 0xAD]).decode('utf-8'): 'CD',
        bytes([0xf0, 0x9f, 0x84, 0xAE]).decode('utf-8'): 'Wz',
        bytes([0xf0, 0x9f, 0x85, 0x8A]).decode('utf-8'): 'HV',
        bytes([0xf0, 0x9f, 0x85, 0x8B]).decode('utf-8'): 'MV',
        bytes([0xf0, 0x9f, 0x85, 0x8C]).decode('utf-8'): 'SD',
        bytes([0xf0, 0x9f, 0x85, 0x8D]).decode('utf-8'): 'SS',
        bytes([0xf0, 0x9f, 0x85, 0x8E]).decode('utf-8'): 'PPV',
        bytes([0xf0, 0x9f, 0x85, 0x8F]).decode('utf-8'): 'WC',
        bytes([0xf0, 0x9f, 0x85, 0xAA]).decode('utf-8'): 'MC',
        bytes([0xf0, 0x9f, 0x85, 0xAB]).decode('utf-8'): 'MD',
        bytes([0xf0, 0x9f, 0x86, 0x8A]).decode('utf-8'): 'P',
        bytes([0xf0, 0x9f, 0x86, 0x8B]).decode('utf-8'): 'IC',
        bytes([0xf0, 0x9f, 0x86, 0x8C]).decode('utf-8'): 'PA',
        bytes([0xf0, 0x9f, 0x86, 0x8D]).decode('utf-8'): 'SA',
        bytes([0xf0, 0x9f, 0x86, 0x8E]).decode('utf-8'): 'AB',
        bytes([0xf0, 0x9f, 0x86, 0x8F]).decode('utf-8'): 'WC',
        bytes([0xf0, 0x9f, 0x86, 0x90]).decode('utf-8'): 'DJ',
        bytes([0xf0, 0x9f, 0x86, 0x91]).decode('utf-8'): 'CL',
        bytes([0xf0, 0x9f, 0x86, 0x92]).decode('utf-8'): 'COOL',
        bytes([0xf0, 0x9f, 0x86, 0x93]).decode('utf-8'): 'FREE',
        bytes([0xf0, 0x9f, 0x86, 0x94]).decode('utf-8'): 'ID',
        bytes([0xf0, 0x9f, 0x86, 0x95]).decode('utf-8'): 'NEW',
        bytes([0xf0, 0x9f, 0x86, 0x96]).decode('utf-8'): 'NG',
        bytes([0xf0, 0x9f, 0x86, 0x97]).decode('utf-8'): 'OK',
        bytes([0xf0, 0x9f, 0x86, 0x98]).decode('utf-8'): 'SOS',
        bytes([0xf0, 0x9f, 0x86, 0x99]).decode('utf-8'): 'UP!',
        bytes([0xf0, 0x9f, 0x86, 0x9A]).decode('utf-8'): 'VS',
        bytes([0xf0, 0x9f, 0x86, 0x9B]).decode('utf-8'): '3D',
        bytes([0xf0, 0x9f, 0x86, 0x9C]).decode('utf-8'): '3D',
        bytes([0xf0, 0x9f, 0x86, 0x9D]).decode('utf-8'): '2K',
        bytes([0xf0, 0x9f, 0x86, 0x9E]).decode('utf-8'): '4K',
        bytes([0xf0, 0x9f, 0x86, 0x9F]).decode('utf-8'): '8K',
        bytes([0xf0, 0x9f, 0x86, 0xA0]).decode('utf-8'): '5.1',
        bytes([0xf0, 0x9f, 0x86, 0xA1]).decode('utf-8'): '7.1',
        bytes([0xf0, 0x9f, 0x86, 0xA2]).decode('utf-8'): '22.2',
        bytes([0xf0, 0x9f, 0x86, 0xA3]).decode('utf-8'): '60P',
        bytes([0xf0, 0x9f, 0x86, 0xA4]).decode('utf-8'): '120P',
        bytes([0xf0, 0x9f, 0x86, 0xA5]).decode('utf-8'): 'd',
        bytes([0xf0, 0x9f, 0x86, 0xA6]).decode('utf-8'): 'HC',
        bytes([0xf0, 0x9f, 0x86, 0xA7]).decode('utf-8'): 'HDR',
        bytes([0xf0, 0x9f, 0x86, 0xA8]).decode('utf-8'): 'Hi-Res',
        bytes([0xf0, 0x9f, 0x86, 0xA9]).decode('utf-8'): 'Lossless',
        bytes([0xf0, 0x9f, 0x86, 0xAA]).decode('utf-8'): 'SHV',
        bytes([0xf0, 0x9f, 0x86, 0xAB]).decode('utf-8'): 'UHD',
        bytes([0xf0, 0x9f, 0x86, 0xAC]).decode('utf-8'): 'VOD',
    }

    b3, b2, b1, b0 = (0xf0, 0x9f, 0x84, 0x80)
    for n in range(10):
        charmap[bytes([b3, b2, b1, b0]).decode('utf-8')] = '{},'.format(n)
        b0 += 1

    b3, b2, b1, b0 = (0xf0, 0x9f, 0x84, 0x90)
    alphas = [[0xf0, 0x9f, 0x84, 0xB0], [0xf0, 0x9f, 0x85, 0x90], [0xf0, 0x9f, 0x85, 0xB0], [0xf0, 0x9f, 0x87, 0xA6]]
    for l in capital_alphabet:
        charmap[bytes([b3, b2, b1, b0]).decode('utf-8')] = '({})'.format(l)
        b0 += 1
        for i, code in enumerate(alphas):
            charmap[bytes(code).decode('utf-8')] = l
            code[3] += 1
            if code[3] == 0xc0:
                code[3] = 0x80
                code[2] += 1

    return charmap


def _build_extended_map():
    charmap = {
        '\u1D00': 'A', '\u1D03': 'B', '\u1D04': 'C', '\u1D05': 'D',
        '\u1D07': 'E', '\u1D0A': 'J', '\u1D0B': 'K', '\u1D0C': 'L',
        '\u1D0D': 'M', '\u1D0F': 'O', '\u1D18': 'K', '\u1D1B': 'T',
        '\u1D1C': 'U', '\u1D20': 'V', '\u1D21': 'W', '\u1D22': 'Z',
        '\u1D29': 'P', '\u1D2C': 'A', '\u1D2E': 'B', '\u1D30': 'D',
        '\u1D31': 'E', '\u1D33': 'G', '\u1D34': 'H', '\u1D35': 'I',
        '\u1D36': 'J', '\u1D37': 'K', '\u1D38': 'L', '\u1D39': 'M',
        '\u1D3A': 'N', '\u1D3C': 'O', '\u1D3E': 'P', '\u1D3F': 'R',
        '\u1D40': 'T', '\u1D41': 'U', '\u1D42': 'W', '\u1D43': 'a',
        '\u1D45': 'a', '\u1D47': 'b', '\u1D48': 'd', '\u1D49': 'e',
        '\u1D4B': 'e', '\u1D4D': 'g', '\u1D4F': 'k', '\u1D50': 'm',
        '\u1D52': 'o', '\u1D56': 'p', '\u1D57': 't', '\u1D58': 'u',
        '\u1D5B': 'v', '\u1D63': 'r', '\u1D64': 'u', '\u1D65': 'v',
        '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0396': 'Z',
        '\u0397': 'H', '\u0399': 'I', '\u039A': 'K', '\u039C': 'M',
        '\u039D': 'N', '\u039F': 'O', '\u03A1': 'P', '\u03A4': 'T',
        '\u03A5': 'Y', '\u03A7': 'X', '\u03BA': 'k', '\u03BD': 'v',
        '\u03BF': 'o', '\u03C7': 'X', '\u03DC': 'F', '\u03DD': 'F',
        '\u03E6': 'b', '\u03E4': 'q', '\u03E5': 'q', '\u03F2': 'c',
        '\u03F3': 'j', '\u03F9': 'C', '\u03FA': 'M', '\u03FB': 'c',
        '\u0251': 'a', '\u0256': 'd', '\u0257': 'd', '\u0260': 'g',
        '\u0261': 'g', '\u0262': 'G', '\u0266': 'h', '\u0267': 'h',
        '\u0268': 'i', '\u026A': 'I', '\u026B': 'l', '\u026C': 'l',
        '\u026D': 'l', '\u0271': 'm', '\u0272': 'n', '\u0273': 'n',
        '\u0274': 'N', '\u027C': 'r', '\u027D': 'r', '\u028D': 'm',
        '\u027E': 'r', '\u0280': 'R', '\u0282': 's', '\u0287': 't',
        '\u0288': 't', '\u0289': 'u', '\u028A': 'u', '\u028B': 'v',
        '\u028F': 'Y', '\u0290': 'z', '\u0291': 'z', '\u0297': 'C',
        '\u0299': 'B', '\u029C': 'H', '\u029D': 'j', '\u029F': 'L',
        '\u025B': 'e', '\u026E': 'b', '\u0284': 'f',
        '\u0531': 'u', '\u0532': 'f', '\u0533': 'q',
        '\u0535': 't', '\u0537': 't', '\u053A': 'd', '\u053B': 'r',
        '\u053C': 'L', '\u0544': 'U', '\u0548': 'n', '\u054D': 'U',
        '\u054F': 'S', '\u0550': 'r', '\u0555': 'O', '\u0556': 'S',
        '\u0559': "'", '\u055A': "'", '\u055B': '`', '\u055D': '`',
        '\u0562': "f", '\u0563': "q", '\u0564': 'n', '\u0565': 't',
        '\u0566': "q", '\u0567': "t", '\u0569': 'p', '\u056A': 'd',
        '\u056B': "h", '\u056C': "l", '\u0570': 'h', '\u0572': 'n',
        '\u0574': "u", '\u0575': "J", '\u0576': 'u', '\u0577': '2',
        '\u0578': "n", '\u057C': "n", '\u057D': 'u', '\u0580': 'n',
        '\u0581': "g", '\u0582': "L", '\u0584': 'p', '\u0585': 'o',
        '\u0586': 'S', '\u0587': "u", '\u0589': ":",
        '\u00DE': 'p', '\u00E0': 'a', '\u00E1': 'a', '\u00E2': 'a',
        '\u00E3': 'a', '\u00E4': 'a', '\u00E5': 'a', '\u00DF': 'b',
        '\u00E8': 'e', '\u00E9': 'e', '\u00EA': 'e', '\u00EB': 'e',
        '\u00EC': 'i', '\u00ED': 'i', '\u00EE': 'i', '\u00EF': 'i',
        '\u00FE': 'b',
        '\u0180': "b", '\u0181': "B", '\u0182': "b", '\u0183': "b",
        '\u0184': "b", '\u0185': "b", '\u0187': "C", '\u0188': "c",
        '\u0189': "D", '\u018A': "D", '\u0190': "e", '\u0191': "F",
        '\u0192': "f", '\u0193': "G", '\u0196': "I", '\u0197': "I",
        '\u0198': "K", '\u0199': "k", '\u019A': "l", '\u019D': "N",
        '\u019E': "n", '\u019F': "O", '\u01A0': "O", '\u01A1': "o",
        '\u01A4': "P", '\u01A5': "p", '\u01A6': "R", '\u01AC': "T",
        '\u01AD': "t", '\u01AE': "T", '\u01AF': "U", '\u01B0': "u",
        '\u01B1': "U", '\u01B2': "V", '\u01B3': "Y", '\u01B4': "y",
        '\u01B5': "Z", '\u01B6': "z", '\u01BC': "5", '\u01BD': "5",
        '\u01BB': "2", '\u01BA': "3", '\u01C3': "!", '\u01C4': "DZ",
        '\u01C5': "Dz", '\u01C6': "dz", '\u01C7': "LJ", '\u01C8': "Lj",
        '\u01C9': "lj", '\u01CA': "NJ", '\u01CB': "Nj", '\u01CC': "nj",
        '\u01CD': "A", '\u01CE': "a", '\u01CF': "I", '\u01D0': "i",
        '\u01D1': "O", '\u01D2': "o", '\u01D3': "U", '\u01D4': "u",
        '\u01D5': "U", '\u01D6': "u", '\u01D7': "U", '\u01D8': "u",
        '\u01D9': "U", '\u01DA': "u", '\u01DB': "U", '\u01DC': "u",
        '\u01DE': "A", '\u01DF': "a", '\u01E0': "A", '\u01E1': "a",
        '\u01E4': "G", '\u01E5': "g", '\u01E6': "G", '\u01E7': "g",
        '\u01E8': "K", '\u01E9': "k", '\u01EA': "Q", '\u01EB': "q",
        '\u01EC': "Q", '\u01ED': "q", '\u01F0': "j", '\u01F1': "DZ",
        '\u01F2': "Dz", '\u01F3': "dz", '\u01F4': "G", '\u01F5': "g",
        '\u01F6': "H", '\u01F8': "N", '\u01F9': "n", '\u01FA': "A",
        '\u01FB': "a", '\u01FE': "O", '\u01FF': "o", '\u0200': "A",
        '\u0201': "a", '\u0202': "A", '\u0203': "a", '\u0204': "E",
        '\u0205': "e", '\u0206': "E", '\u0207': "e", '\u0208': "I",
        '\u0209': "i", '\u020A': "I", '\u020B': "i", '\u020C': "O",
        '\u020D': "o", '\u020E': "O", '\u020F': "o", '\u0210': "R",
        '\u0211': "r", '\u0212': "R", '\u0213': "r", '\u0214': "U",
        '\u0215': "u", '\u0216': "U", '\u0217': "u", '\u0218': "S",
        '\u0219': "s", '\u021A': "T", '\u021B': "t", '\u021E': "H",
        '\u021F': "h", '\u0220': "n", '\u0221': "d", '\u0222': "8",
        '\u0223': "8", '\u0224': "Z", '\u0225': "z", '\u0226': "A",
        '\u0227': "a", '\u0228': "E", '\u0229': "e", '\u022A': "O",
        '\u022B': "o", '\u022C': "O", '\u022D': "o", '\u022E': "O",
        '\u022F': "o", '\u0230': "O", '\u0231': "o", '\u0232': "Y",
        '\u0233': "y", '\u0234': "l", '\u0235': "n", '\u0236': "t",
        '\u0237': "j", '\u023A': "A", '\u023B': "C", '\u023C': "c",
        '\u023D': "t", '\u023E': "T", '\u023F': "s", '\u0240': "z",
        '\u0241': "?", '\u0242': "?", '\u0243': "B", '\u0244': "U",
        '\u0245': "A", '\u0246': "E", '\u0247': "e", '\u0248': "J",
        '\u0249': "j", '\u024A': "q", '\u024B': "q", '\u024C': "R",
        '\u024D': "r", '\u024E': "Y", '\u024F': "y",
        '\u04C3': "K", '\u04C4': "k",
    }

    return charmap


def get_table():
    charmap = {}

    charmap.update(_build_halfwidth())
    charmap.update(_build_enclosed_supplements())
    charmap.update(_build_extended_map())

    return charmap


if __name__ == "__main__":
    get_table()
