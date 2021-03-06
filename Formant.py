import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy
import librosa
from sys import exit

# Getting sound file name, window size, hoplength, window type

filename= input('enter file name:')
winsize =int(input('Enter Window Size(in millisec):'))
hoplength =int(input('Enter Hop length(in millisec):'))
wintype =input('Enter window type(hamm or rect):')

# reading the wav file and sampling frequency and plotting it

d, fs = sf.read(f'{filename}.wav')
print('sampling rate=', fs)
ln = len(d)
n = np.linspace(0, ln-1, ln)
plt.figure(1)
plt.plot(n,d)
plt.legend(['plot of sample'])
plt.xlabel('sample')
plt.ylabel('amplitude')
plt.grid()

# Framing the sound file using windowing technique using enframe function

def enframe(x, winsize, hoplength, fs, wintype):
    hpln = int(fs*hoplength*0.001)
    wnsz = int(fs*winsize*0.001)
    temp = wnsz-(len(x) % hpln)
    z = np.pad(x, (0, temp), 'constant')
    if wintype == 'hamm':
        win = np.hamming(wnsz)
    elif wintype == 'rect':
        win = np.ones(wnsz)
    frame = []
    l = len(x)
    for i in range(0, l, hpln):
        a = z[i:i+wnsz]*win
        frame.append(a)
    return(frame)

windowdata = enframe(d, winsize, hoplength, fs, wintype)

# taking log magnitude spectrum and peak picking of smoothened the spectrum using LPC analysis for each frame

X = [];F = [];L = []; P = []
for i in windowdata:
    b = np.log10(np.abs(np.fft.fft(i)))
    n1 = np.linspace(0, len(b)-1, len(b))
    b = b[0:len(b)//2]
    freq = (fs/1000)*n1*(1/len(n1))
    freq = freq[0:len(freq)//2]
    X.append(b)
    F.append(freq)
    a = librosa.lpc(i,int((fs/1000)+2))
    _ ,h = scipy.signal.freqz([1], a,worN= len(freq))
    y = np.log10(np.abs(h))
    peaks, _ = find_peaks(y)
    P.append(peaks)
    x= np.mean(y[peaks])        
    y=y-x
    L.append(y)
    
Formants=[]
for k in range(len(F)):
    p=P[k]
    f=F[k]
    Formants.append(f[p])
     
with open(f'{filename}.csv', 'w', newline='') as csvfile:
    fieldnames = ['frame', 'peak frequencies']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(Formants)):
        writer.writerow({'frame':f'Frame {i+1}  ' , 'peak frequencies':f'   {Formants[i]}'})   
    
# Plotting all the log magnitude plot with peak detection for any given frame
i=1;
while(True):
    fram=input(f'Enter frame number from 1 to {len(X)} or to stop just press enter:')
    if fram=='':
        exit()
    else:
        j=(int(fram)-1)
        print(f'Formants are(in kilo hertz): {Formants[j]}')
        plt.figure(i+1)
        b=windowdata[j]
        a = np.linspace(0,len(b)-1, len(b))
        plt.subplot(4,1,1)
        plt.plot(a,b)
        plt.title(f'plot of win farme:{j+1}')
        plt.xlabel('sample')
        plt.ylabel('amplitude')
        plt.grid()
        plt.figure(i+2)
        f=F[j]
        x=X[j]
        p=P[j]
        l=L[j]
        plt.plot(f,x)
        plt.plot(f,l)
        plt.plot(f[p], l[p], "ob") 
        plt.xlabel(r'freq in KHz units')
        plt.ylabel('log mag in db')
        plt.title(f'spectrum of win frame :{j+1}')
        plt.grid()
        i+=2
