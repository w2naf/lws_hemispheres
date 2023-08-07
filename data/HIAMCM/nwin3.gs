'reinit'
'set display color white'
'clear'

'set xlopts 1 4 0.12'
'set ylopts 1 4 0.12'
'set clopts 1 4 0.12'
'set annot  1 4'

'run ede.col'

'open 07DEC2018-16MAR2019.mzgw.ctl'

'set t 1 100  '

'set lev 0 375 '   
"set lat 60"
'run ede.col'

* x-momen. deposition
'set vpage  0.5 10.5  3.6   7.4 '
'set parea  1.0 9.5   0.7   3.7'
'set gxout shaded'
'set grads off'
'set ccols  19   4   11   5    3  0  7  12  8   2   6 '
'set clevs     -80 -40  -20  -10 -5  5  10  20  40  80 '
'set grads off'
'd ave(dragx,lat=50,lat=70)  '
*xl = 9.6; xwid = 0.2 ; yb = 0.710 ; ywid = 0.229
xl = 9.6; xwid = 0.2 ; yb = 0.710 ; ywid = 0.270
dummy = vcbar (xl,xwid,yb,ywid)

* average GW vert. wind perturbation'
'set vpage  0.5 10.5  0.2   4.0 '
'set parea  1.0 9.5   0.6   3.6'
'set ccols   4   11   5   13  3  10   7  12   8    2  6  '
'set clevs     0.5  1  1.5  2  2.5  3  4    6   8   10   '
'set grads off'
'set gxout shaded'
'd sqrt( ave(ww,lat=50,lat=90) ) '
xl = 9.6; xwid = 0.2 ; yb = 0.610 ; ywid = 0.270 ;
*xl = 9.6; xwid = 0.2 ; yb = 0.610 ; ywid = 0.298 ;
dummy = vcbar (xl,xwid,yb,ywid)
*
"set parea off"
"set vpage off"

'set string 1 c 4 0'
'set strsiz 0.12'

'set string 1 c 4 90'
'draw string 0.74  5.67 altitude (km)'
'draw string 0.74  2.22 altitude (km)'

'set string 1 c 3  0  '
'set strsiz 0.13'
"draw string 5.3  7.13 (a) 156`a `nkm`a `n<`a `n`3l`0`bH `n<`a `n1350`a `nkm: x-momentum deposition (ms`a-1`n), 50`ao`nN-70`ao`nN"
"draw string 5.3  3.63 (b) 156`a `nkm`a `n<`a `n`3l`0`bH `n<`a `n1350`a `nkm: average vertical wind variation (ms`a-1`n), 50`ao`nN-90`ao`nN"

'gxprint nwin3.eps'
'gxprint nwin3.png'

********************************************************************
function vcbar (xl,xwid,yb,ywid)  (E. Becker 14.10.2004)
*  Check shading information
'query shades'
shdinfo = result
*say shdinfo
if (subwrd(shdinfo,1)='None')
  say 'Cannot plot color bar: No shading information'
  return
endif
cnum = subwrd(shdinfo,5)
*  Plot colorbar
'set string 1 c 4'
'set strsiz 0.095 0.095'
num   = 0
x1 = xl
y1 = yb
x2 = xl+xwid
while (num<cnum)
  rec = sublin(shdinfo,num+2)
  col = subwrd(rec,1)
  hi  = subwrd(rec,3)
  'set line 'col
  y2 = y1 + ywid
  'draw recf 'x1' 'y1' 'x2' 'y2
  if (num<cnum-1 )
*    if( num=0 | num=2 | num=4 | num=6 | num=8 | num=10 | num=12 | num=14 )
*      'draw string '%(xl+xwid+0.3)%' 'y2' 'hi
*    endif
    'draw string '%(xl+xwid+0.18)%' 'y2' 'hi
  endif
  num = num + 1
  y1 = y2
endwhile
'set line 1'
x1=xl; y1=yb
'draw line 'x1' 'y1' 'x1' 'y2
'draw line 'x1' 'y2' 'x2' 'y2
'draw line 'x2' 'y1' 'x2' 'y2
'draw line 'x1' 'y1' 'x2' 'y1
return
*******************************************************************
