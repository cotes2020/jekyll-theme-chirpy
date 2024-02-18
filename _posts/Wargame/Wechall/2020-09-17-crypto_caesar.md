---
title : "Wechall - Caesar I"
categories : [Wargame, Wechall]
---

## Crypto - Caesar I
<hr style="border-top: 1px solid;"><br>

```MAX JNBVD UKHPG YHQ CNFIL HOXK MAX ETSR WHZ HY VTXLTK TGW RHNK NGBJNX LHENMBHG BL KHLVVIWZXHGV```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

```python
import string

cipher=list("MAX JNBVD UKHPG YHQ CNFIL HOXK MAX ETSR WHZ HY VTXLTK TGW RHNK NGBJNX LHENMBHG BL KHLVVIWZXHGV")

for i in range(1,26):
    plain=''
    for j in cipher:
        if j != ' ' :
            shift=ord(j)+i
            if shift <= ord('Z') :
                plain+=chr(shift)
            else :
                plain+= chr(shift - 26)
        else :
            plain+=j       
    print("plaintext:",plain, ", key :",i)
```

<br>

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
	char ciphertext[] = "MAX JNBVD UKHPG YHQ CNFIL HOXK MAX ETSR WHZ HY VTXLTK TGW RHNK NGBJNX LHENMBHG BL KHLVVIWZXHGV";
	int decryptkey=1;
	int len = strlen(ciphertext);
	char *plaintext=(char*)malloc(len*sizeof(char));

	while (decryptkey < 26) {
		for (int i = 0; i < len; i++) {
			if (ciphertext[i] != ' ') {
				char shift = ciphertext[i] + decryptkey;
				if (shift <= 'Z') {
					plaintext[i] = shift;
				}
				else {
					plaintext[i] = shift - 26;
				}
			}
			else plaintext[i] = ciphertext[i];
		}
		printf("plaintext : ");
		for (int i = 0; i < len; i++) {
			printf("%c", plaintext[i]);
		}
		printf(" -> key : %d\n", decryptkey);
		decryptkey++;
	}

	free(plaintext);
	return 0;
}

''' Result
plaintext : NBY KOCWE VLIQH ZIR DOGJM IPYL NBY FUTS XIA IZ WUYMUL UHX SIOL OHCKOY MIFONCIH CM LIMWWJXAYIHW -> key : 1
plaintext : OCZ LPDXF WMJRI AJS EPHKN JQZM OCZ GVUT YJB JA XVZNVM VIY TJPM PIDLPZ NJGPODJI DN MJNXXKYBZJIX -> key : 2
plaintext : PDA MQEYG XNKSJ BKT FQILO KRAN PDA HWVU ZKC KB YWAOWN WJZ UKQN QJEMQA OKHQPEKJ EO NKOYYLZCAKJY -> key : 3
plaintext : QEB NRFZH YOLTK CLU GRJMP LSBO QEB IXWV ALD LC ZXBPXO XKA VLRO RKFNRB PLIRQFLK FP OLPZZMADBLKZ -> key : 4
plaintext : RFC OSGAI ZPMUL DMV HSKNQ MTCP RFC JYXW BME MD AYCQYP YLB WMSP SLGOSC QMJSRGML GQ PMQAANBECMLA -> key : 5
plaintext : SGD PTHBJ AQNVM ENW ITLOR NUDQ SGD KZYX CNF NE BZDRZQ ZMC XNTQ TMHPTD RNKTSHNM HR QNRBBOCFDNMB -> key : 6
plaintext : THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG OF CAESAR AND YOUR UNIQUE SOLUTION IS ROSCCPDGEONC -> key : 7 <-- Here -->
plaintext : UIF RVJDL CSPXO GPY KVNQT PWFS UIF MBAZ EPH PG DBFTBS BOE ZPVS VOJRVF TPMVUJPO JT SPTDDQEHFPOD -> key : 8
plaintext : VJG SWKEM DTQYP HQZ LWORU QXGT VJG NCBA FQI QH ECGUCT CPF AQWT WPKSWG UQNWVKQP KU TQUEERFIGQPE -> key : 9
plaintext : WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ RI FDHVDU DQG BRXU XQLTXH VROXWLRQ LV URVFFSGJHRQF -> key : 10
plaintext : XLI UYMGO FVSAR JSB NYQTW SZIV XLI PEDC HSK SJ GEIWEV ERH CSYV YRMUYI WSPYXMSR MW VSWGGTHKISRG -> key : 11
plaintext : YMJ VZNHP GWTBS KTC OZRUX TAJW YMJ QFED ITL TK HFJXFW FSI DTZW ZSNVZJ XTQZYNTS NX WTXHHUILJTSH -> key : 12
plaintext : ZNK WAOIQ HXUCT LUD PASVY UBKX ZNK RGFE JUM UL IGKYGX GTJ EUAX ATOWAK YURAZOUT OY XUYIIVJMKUTI -> key : 13
plaintext : AOL XBPJR IYVDU MVE QBTWZ VCLY AOL SHGF KVN VM JHLZHY HUK FVBY BUPXBL ZVSBAPVU PZ YVZJJWKNLVUJ -> key : 14
plaintext : BPM YCQKS JZWEV NWF RCUXA WDMZ BPM TIHG LWO WN KIMAIZ IVL GWCZ CVQYCM AWTCBQWV QA ZWAKKXLOMWVK -> key : 15
plaintext : CQN ZDRLT KAXFW OXG SDVYB XENA CQN UJIH MXP XO LJNBJA JWM HXDA DWRZDN BXUDCRXW RB AXBLLYMPNXWL -> key : 16
plaintext : DRO AESMU LBYGX PYH TEWZC YFOB DRO VKJI NYQ YP MKOCKB KXN IYEB EXSAEO CYVEDSYX SC BYCMMZNQOYXM -> key : 17
plaintext : ESP BFTNV MCZHY QZI UFXAD ZGPC ESP WLKJ OZR ZQ NLPDLC LYO JZFC FYTBFP DZWFETZY TD CZDNNAORPZYN -> key : 18
plaintext : FTQ CGUOW NDAIZ RAJ VGYBE AHQD FTQ XMLK PAS AR OMQEMD MZP KAGD GZUCGQ EAXGFUAZ UE DAEOOBPSQAZO -> key : 19
plaintext : GUR DHVPX OEBJA SBK WHZCF BIRE GUR YNML QBT BS PNRFNE NAQ LBHE HAVDHR FBYHGVBA VF EBFPPCQTRBAP -> key : 20
plaintext : HVS EIWQY PFCKB TCL XIADG CJSF HVS ZONM RCU CT QOSGOF OBR MCIF IBWEIS GCZIHWCB WG FCGQQDRUSCBQ -> key : 21
plaintext : IWT FJXRZ QGDLC UDM YJBEH DKTG IWT APON SDV DU RPTHPG PCS NDJG JCXFJT HDAJIXDC XH GDHRRESVTDCR -> key : 22
plaintext : JXU GKYSA RHEMD VEN ZKCFI ELUH JXU BQPO TEW EV SQUIQH QDT OEKH KDYGKU IEBKJYED YI HEISSFTWUEDS -> key : 23
plaintext : KYV HLZTB SIFNE WFO ALDGJ FMVI KYV CRQP UFX FW TRVJRI REU PFLI LEZHLV JFCLKZFE ZJ IFJTTGUXVFET -> key : 24
plaintext : LZW IMAUC TJGOF XGP BMEHK GNWJ LZW DSRQ VG
'''
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
