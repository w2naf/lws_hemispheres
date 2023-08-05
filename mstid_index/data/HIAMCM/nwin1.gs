'reinit'
'set display color white'
'clear'

'set xlopts 1 4 0.12'
'set ylopts 1 4 0.12'
'set clopts 1 4 0.12'
'set annot  1 4'

'run ede.col'

'open  07DEC2018-16MAR2019.mzgw.ctl'
'set t 1 100 '

'set lev  0 375 '  
"set lat 60"
'run ede.col'

* (1) T                         
'set vpage  0.5 10.5  3.6   7.4 '
'set parea  1.0 9.5   0.7   3.7'
'set gxout shaded'
'set ccols   4   11   5   13  3  10   7  12   8    2    6  32  '
*'set clevs    180 200  220 240 260 280 320 400 550  700  900 '
'set clevs    180 200  220 240 260 280 320 400 500  600  700 '
*xl = 9.6; xwid = 0.2 ; yb = 0.710 ; ywid = 0.23
xl = 9.6; xwid = 0.2 ; yb = 0.710 ; ywid = 0.247
'set grads off'
'd ave(T,lat=70,lat=90) '      
dummy = vcbar (xl,xwid,yb,ywid)

* (2) u
*
'set vpage  0.5 10.5  0.2   4.0 '
'set parea  1.0 9.5   0.6   3.6'
*'set ccols   19  9   14  4   11   5   13  3  10   7  12  8    2    6 '
'set ccols   11   5    3  0  10  7  12  8   2  6  32   '
'set clevs    -20  -10 -5  5  10  20  30  50 70 90  '
'set grads off'
'd ave(u,lat=50,lat=70)'
*xl = 9.6; xwid = 0.2 ; yb = 0.610 ; ywid = 0.247 ;
xl = 9.6; xwid = 0.2 ; yb = 0.610 ; ywid = 0.272 ;
dummy = vcbar (xl,xwid,yb,ywid)
*
"set parea off"
"set vpage off"


'set string 1 c 4 0'
'set strsiz 0.12'
*'draw string  5.3 0.38 `2UT (day of July)'

'set string 1 c 4 90'
'draw string 0.74  5.67 altitude (km)'
'draw string 0.74  2.22 altitude (km)'

'set string 1 c 3  0  '
'set strsiz 0.13'
"draw string 5.3  7.13 (a) temperature (K), 70`ao`nN-90`ao`nN"
"draw string 5.3  3.63 (b) zonal wind (ms`a-1`n), 50`ao`nN-70`ao`nN"


'gxprint nwin1.eps'
'gxprint nwin1.png'


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
