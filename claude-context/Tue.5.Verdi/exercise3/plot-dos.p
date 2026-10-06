set terminal pdfcairo dashed color enhanced font 'Arial,21' linewidth 1.5 size 6,4
set key top right Left reverse samplen 1.4
THz2mev=4.13567
set xlabel '{/Symbol w} (meV)'
set ylabel 'v-DOS (arb. un.)'
unset ytics #set mytics 2; set ytics scale 1,0.8
set output 'plot-dos.pdf'
set xr [0:180]; set yr [0:27]
plot 'model_I-total_dos.dat' u ($1*THz2mev):2 w l lc rgb 'black' lw 2 tit 'model I', \
'model_II-total_dos.dat' u ($1*THz2mev):($2*2) w l lc rgb 'black' lw 2 lt 3 tit 'model II', \
'model_III-total_dos.dat' u ($1*THz2mev):($2*2) w l lc rgb 'black' lw 2 lt 2 tit 'model III'

