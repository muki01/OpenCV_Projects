import cv2
from cvzone.HandTrackingModule import HandDetector

class Buton:
    def __init__(self,konum,genislik,yukseklik,deger):

        self.konum = konum
        self.genislik = genislik
        self.yukseklik = yukseklik
        self.deger = deger

    def ciz(self,img):

        cv2.rectangle(img, self.konum, (self.konum[0]+self.genislik, self.konum[1]+self.yukseklik),
                      (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, self.konum, (self.konum[0] + self.genislik, self.konum[1] + self.yukseklik),
                     (50, 50, 50), 3)

        cv2.putText(img, self.deger, (self.konum[0] + 40, self.konum[1] + 60), cv2.FONT_HERSHEY_PLAIN,
                    2, (50, 50, 50), 2)

    def klikKontrol(self,x,y):
        if self.konum[0] < x <self.konum[0] + self.genislik and \
                self.konum[1] < y < self.konum[1] + self.yukseklik:

            cv2.rectangle(img, self.konum, (self.konum[0] + self.genislik, self.konum[1] + self.yukseklik),
                          (255, 255, 255), cv2.FILLED)
            cv2.rectangle(img, self.konum, (self.konum[0] + self.genislik, self.konum[1] + self.yukseklik),
                          (50, 50, 50), 3)
            cv2.putText(img, self.deger, (self.konum[0] + 25, self.konum[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5)
            return True
        else:
            return False

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Buton oluşturma

butonListeDegerleri = [['7', '8', '9', '*'],
                       ['4', '5', '6', '='],
                       ['1', '2', '3', '+'],
                       ['0', '/', '.', '-'],]

butonListesi = []
for x in range(4):
    for y in range(4):
        xkonumu = x*100 + 700
        ykonumu = y*100 + 150
        butonListesi.append(Buton((xkonumu,ykonumu),100,100,butonListeDegerleri[y][x]))

# Değişkenler
denklemim = ''
gecikmeSayaci = 0

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Elin algılanması
    hands, img = detector.findHands(img, flipType=False)

    # Butonları çiz
    cv2.rectangle(img, (700,50), (700 + 400, 70 + 100),
                  (225, 225, 225), cv2.FILLED)
    cv2.rectangle(img, (700,50), (700 + 400, 70 + 100),
                  (50, 50, 50), 3)

    for buton in butonListesi:
        buton.ciz(img)

    # Eli kontrol et
    if hands:
        lmList = hands[0]['lmList']
        length, _, img = detector.findDistance(lmList[8][0:2],lmList[4][0:2],img)
        #print(length)
        x,y = lmList[4][0:2]
        if length<40:
            for i, buton in enumerate(butonListesi):
                if buton.klikKontrol(x,y) and gecikmeSayaci == 0:
                    mevcutDeger = (butonListeDegerleri[int(i%4)][int(i/4)])
                    if mevcutDeger == '=':
                        denklemim = str(eval(denklemim))   # eval kullanıldı
                    else:
                        denklemim += mevcutDeger
                    gecikmeSayaci = 1  #

    # Çoklu göstermeleri önleme
    if gecikmeSayaci != 0:
        gecikmeSayaci += 1
        if gecikmeSayaci > 10:
            gecikmeSayaci = 0

    # Sonucu gösterme
    cv2.putText(img, denklemim, (710, 120), cv2.FONT_HERSHEY_PLAIN,
                3, (50, 50, 50), 3)

    # Display
    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('c'):
        denklemim = ''

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
