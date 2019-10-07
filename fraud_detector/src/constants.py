import os

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other',
          'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo',
          'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft',
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google',
          'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other',
          'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other',
          'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other',
          'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink',
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo',
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
          'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other',
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com':
              'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol',
          'juno.com': 'other', 'icloud.com': 'apple'}

dev_info_map = {'Windows': 'Windows', 'iOS Device': 'iOS Device', 'MacOS': 'MacOS', 'Trident/7.0': 'Trident/7.0'}


INPUT_DIR = '../data/input_data/ieee-fraud-detection/'
OUTPUT_DIR = '../data/output_data/'

vesta = [f"V{k}" for k in range(12, 339)]

transaction_features = ['TransactionID', 'isFraud']
transaction_features.extend(vesta)
transaction_features.extend(['ProductCD', 'TransactionDT', 'TransactionAmt', 'dist1', 'dist2',
                             'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                             'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',
                             'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
                             ]
                            )
counting_variables_numeric = ['C1', 'C2', 'C5', 'C6', 'C9', 'C11', 'C13', 'C14']
counting_variables_categoric = ['C7', 'C8', 'C10', 'C12']+['C3', 'C4']
card_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
m_features = [f"M{k}" for k in range(1, 10)]

time_delta_variables = [f"D{k}" for k in range(1, 16)]

transaction_features.extend(counting_variables_numeric)
transaction_features.extend(counting_variables_categoric)
transaction_features.extend(time_delta_variables)

set_counting_types = {feature: 'int' for feature in counting_variables_categoric}
set_delta_types = {feature: 'float' for feature in time_delta_variables}
set_v_types = {feature: 'float' for feature in vesta}


identity_features = ['TransactionID', 'DeviceInfo', 'DeviceType']+\
                     [f'id_0{k}' for k in range(1, 10)] + \
                     [f'id_{k}' for k in range(10, 21)] + [f'id_{k}' for k in range(28, 39)]

browser_mapper = {"samsung browser 7.0": 1,
                  "opera 53.0": 1,
                  "mobile safari 10.0": 1,
                  "google search application 49.0": 1,
                  "firefox 60.0": 1,
                  "edge 17.0": 1,
                  "chrome 69.0": 1,
                  "chrome 67.0 for android": 1,
                  "chrome 63.0 for android": 1,
                  "chrome 63.0 for ios": 1,
                  "chrome 64.0": 1,
                  "chrome 64.0 for android": 1,
                  "chrome 64.0 for ios": 1,
                  "chrome 65.0": 1,
                  "chrome 65.0 for android": 1,
                  "chrome 65.0 for ios": 1,
                  "chrome 66.0": 1,
                  "chrome 66.0 for android": 1,
                  "chrome 66.0 for ios": 1}

risky_addr1 = [256, 263, 266, 267, 524, 525,
               271, 273, 532, 533, 534, 537,
               538, 281, 285, 287, 288, 289,
               291, 293, 311, 317, 319, 320,
               334, 336, 342, 344, 350, 354,
               355, 357, 103, 362, 107, 108,
               109, 363, 364, 367, 114, 115,
               116, 370, 118, 121, 378, 380,
               383, 388, 135, 136, 392, 138,
               394, 140, 398, 147, 149, 150,
               405, 407, 412, 413, 414, 415,
               419, 165, 421, 422, 423, 169,
               424, 173, 175, 176, 179, 437,
               438, 440, 186, 442, 188, 447,
               192, 449, 197, 455, 460, 461,
               207, 209, 212, 473, 475, 222,
               480, 228, 229, 230, 484, 487,
               490, 495, 240, 497, 246, 510, 539]
risky_addr2 = {11, 12, 33, 37, 41, 42, 45, 53, 56, 58, 64, 67, 80, 81, 85, 90, 91, 95, 99}

addr1_mapper = {idx: new_idx for new_idx, idx in enumerate(risky_addr1)}
addr2_mapper = {idx: new_idx for new_idx, idx in enumerate(risky_addr2)}
