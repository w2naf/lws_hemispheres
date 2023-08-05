c
c reader for post-processed HIMAMCM data *.mzgw.grads
c ( EB / July 2023 ) 
c
      implicit none
c      
      integer nlat  ! number of latitudes
      integer lev1  ! number of model layers
      integer idat  ! number of variables
      parameter(nlat=90,lev1=261,idat=54)
c      
      integer j,lz,len,krec,it,id
c      
      integer ylat(nlat)  ! latitudes
      real*4  altz(lev1)  ! altitudes in km 
      real*4  dat(nlat,lev1,idat)  ! all 54 variables as noted in the
c                                  ! ctl file on a latitude-height grid 
      character ch50*50       
c
c latitude grid
c
      do j = 1,nlat
        ylat(j) = -89 + 2*(j-1) 
      enddo
c     print*,'ylat = ',ylat
c
c vertical grid 
c
c     dz = 0.5 km up to 70 km
      do lz = 1,140
        altz(lz) = 0.5*real( (lz-1) )
      enddo
c     dz = 1 km from 70 to 120 km
      do lz = 1,50
        altz(140+lz) = 70.0  + real( (lz-1) )
      enddo
c     dz = 2 km from 120 to 200 km
      do lz = 1,40
        altz(190+lz) = 120.0 + real( 2*(lz-1) )
      enddo
c     dz = 5 km from 200 to 300 km
      do lz = 1,20
        altz(230+lz) = 200.0 + real( 5*(lz-1) )
      enddo
c     dz = 10 km from 300 to 400 km
      do lz = 1,11
         altz(250+lz) = 300.0 + real( 10*(lz-1) )
      enddo
c     print*,'altz = ', altz
c
c open grads file and read data
c      
      ch50 = '07DEC2018-16MAR2019'
      len=index(ch50,' ')-1
      open(12,file=ch50(1:len)//'.mzgw.grads',status='unknown',
     &     form='unformatted',access='direct',recl=4*nlat)
c
      it = 0
      krec  = 0
 100  it = it + 1
      print*
      print*,'try to read time step, it = ',it 
      do id = 1,idat
        do lz = 1,lev1  
          krec = krec+1
          read(12,rec=krec,err=900)( dat(j,lz,id), j=1,nlat )
        enddo
      enddo 
      print*,' after read it = ',it 
c     write z, p, T, u, v, w profiles over Boulder for
c     the actual time step 
      j = 75
      print*
      print*,'j,ylat(j)=',j,ylat(j) 
      print*
      print*,' lz | z(km) | T(K) | u (m/s) | v\p (m/s) | T\p (K) |',
     &' w\p (m/s) | drag-x (m/s/d) | drag-y (m/s/d) '
c     do lz = lev1,1,-1
      do lz = 250,220,-2
        write(*,200)lz,altz(lz),
     &                int(dat(j,lz,5)),       ! T
     &                int(dat(j,lz,1)),       ! U
     &  int(sqrt(dat(j,lz,25)+dat(j,lz,27))), ! average GW horiz. wind perturbation
     &  int(sqrt(dat(j,lz,33))),              ! average GW T perturbation 
     &  int(sqrt(dat(j,lz,34))),              ! average GW vertical wind perturbation 
     &                int(dat(j,lz,37)),      ! dragx   
     &                int(dat(j,lz,38))       ! dragy   
      enddo   
 200  format(1x,i3,
     &       2x,f5.1, ! altz
     &       5x,i3,   ! T 
     &       3x,i4,   ! U 
     &      10x,i2,   ! average horiz. wind pert. 
     &      10x,i2,   ! average T pert. 
     &       6x,i4,   ! average vert. wind pert. 
     &      10x,i4,   ! dragx 
     &      11x,i4 )  ! dragy 
      pause
      goto 100
c
 900  print*,'number of time steps in the file = ',it-1
c
      stop
      end
