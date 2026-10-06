set terminal pdfcairo dashed color enhanced font 'Arial,21' linewidth 1.5 size 16,4
set style line 1 lt 1 lw 1.5 lc rgb 'gray50'
set style line 2 lt 1 lw 1.5 lc rgb 'medium-blue'
set key horizontal top right Left reverse samplen 1.4 at 0.68,97
THz2mev=4.13567
set ylabel '{/Symbol w} (meV)' offset 1
set mytics 2; set ytics scale 1,0.8
set output 'plot-phonons.pdf'
set multiplot

set size 0.34,1
set lmargin 7
set origin 0,0
x0=0; xf=0.74611780; set xr [x0:xf]
x1=0.17093020; x2=0.36830340; x3=0.46699000; x4=0.53677190
y1=-30; y2=100; set yr [y1:y2]
set xtics ( 'L' x0, '{/Symbol G}' x1, 'X' x2, 'W' x3, 'K' x4, '{/Symbol G}' xf )
set arrow from x1,y1 to x1,y2 nohead lw 0.5 front
set arrow from x2,y1 to x2,y2 nohead lw 0.5 front
set arrow from x3,y1 to x3,y2 nohead lw 0.5 front
set arrow from x4,y1 to x4,y2 nohead lw 0.5 front
set arrow from x0,0 to xf,0 nohead lt 2 lc rgb "black" lw 1
set  label 1 'Cubic' at 0.05,93
plot '../ph-ref/zro2-c-band.dat' u 1:($2*THz2mev) w l ls 1 tit 'DFT', \
'zro2-c-band.dat' u 1:($2*THz2mev) w l ls 2 tit 'MACE'

unset ylabel
set yr [0:100]
set origin 0.33,0
unset arrow
x0=0; xf=0.85144230; set xr [x0:xf]
x1=0.13941340; x2=0.27882670; x3=0.47598700; x4=0.57261560; x5=0.71202900
y1=0; y2=100; set yr [y1:y2]
set xtics ( '{/Symbol G}' x0, 'X' x1, 'M' x2, '{/Symbol G}' x3, 'Z' x4, 'R' x5, 'A' xf )
set arrow from x1,y1 to x1,y2 nohead lw 0.5 front
set arrow from x2,y1 to x2,y2 nohead lw 0.5 front
set arrow from x3,y1 to x3,y2 nohead lw 0.5 front
set arrow from x4,y1 to x4,y2 nohead lw 0.5 front
set arrow from x5,y1 to x5,y2 nohead lw 0.5 front
set  label 1 'Tetragonal' at 0.05,95
plot '../ph-ref/zro2-t-band.dat' u 1:($2*THz2mev) w l ls 1 notit, \
'zro2-t-band.dat' u 1:($2*THz2mev) w l ls 2 notit

set origin 0.65,0
unset arrow
x0=0; xf=0.83917490; set xr [x0:xf]
x1=0.09586810; x2=0.19163620; x3=0.28750420; x4=0.38327230; x5=0.50895160; x6=0.60481970; x7=0.74032690
set xtics ( '{/Symbol G}' x0, 'Z' x1, 'D' x2, 'B' x3, '{/Symbol G}' x4, 'A' x5, 'E' x6, 'Y' x7, '{/Symbol G}' xf )
set arrow from x1,y1 to x1,y2 nohead lw 0.5 front
set arrow from x2,y1 to x2,y2 nohead lw 0.5 front
set arrow from x3,y1 to x3,y2 nohead lw 0.5 front
set arrow from x4,y1 to x4,y2 nohead lw 0.5 front
set arrow from x5,y1 to x5,y2 nohead lw 0.5 front
set arrow from x6,y1 to x6,y2 nohead lw 0.5 front
set arrow from x7,y1 to x7,y2 nohead lw 0.5 front
set  label 1 'Monoclinic' at 0.05,95
plot '../ph-ref/zro2-m-band.dat' u 1:($2*THz2mev) w l ls 1 notit, \
'zro2-m-band.dat' u 1:($2*THz2mev) w l ls 2 notit

