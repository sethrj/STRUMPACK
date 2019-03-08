module A2D

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  2D Helmholtz matrix generation  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  

!use NDpre2D
use partition
implicit none


contains


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Acoustic matrix A  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A2D_acoustic( freq, Nz, Nx, N_delta, sigma0, hz, hx, locHZ, locTZ, locHX, &
         locTX, label, vel, rho, n, nnz, a_ind, a_col, a_val) !A )

        complex(kind=4) :: freq, sigma0, hz, hx
        integer         :: Nz, Nx, N_delta, locHZ, locHX, locTZ, locTX
        integer         :: label(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::   vel(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::   rho(locTZ-locHZ+3,locTX-locHX+3)
        integer :: n, nnz, a_ind(n), a_col(nnz)
        complex(kind=4) :: a_val(nnz)
!        type(CSR)       :: A

        integer                     :: zz, xx, kk, lvl, length, counter, col_tmp(9)
        complex(kind=4)             :: rx, Czz, Cxx, Czx, tmp, tmp1, tmp2, vel_tmp(9)
        complex(kind=4)             :: rho_tmp(9), val_tmp(9)
        complex(kind=4),allocatable :: Sz(:), Sx(:)
        a_ind = 0

        ! the ratio of the spatial step size and parameters
        rx  = hz / hx    

        ! generate Sz and Sx
        allocate(Sz(Nz),Sx(Nx))
        Sz = ONE
        Sx = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function 
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
        end do

        ! allocate a_ind
        length = (locTZ-locHZ+1)*(locTX-locHX+1)    ! the order of the matrix A: number of unknows
 !       allocate(a_ind(length+1))
        a_ind = 0
        
        do xx = locHX,locTX
            do zz = locHZ,locTZ
               lvl = label(zz-locHZ+1,xx-locHX+1) + 1
               if (xx > locHX) then
                   if (zz > locHZ) then
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if
                       a_ind(lvl) = a_ind(lvl) + 1
                   if (zz < locTZ) then
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if
               end if
                   if (zz > locHZ) then
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if                
                       a_ind(lvl) = a_ind(lvl) + 1
                   if (zz < locTZ) then               
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if
                       a_ind(lvl) = a_ind(lvl) + 1
                   if (zz < locTZ) then
                       a_ind(lvl) = a_ind(lvl) + 1
                   end if
               end if
            end do
        end do

        a_ind(1) = 1
        do kk = 1,length
            a_ind(kk+1) = a_ind(kk+1) + a_ind(kk)
        end do

        ! allocate a_col and a_val
        length = 9*(locTZ-locHZ+1)*(locTX-locHX+1) - 6*((locTZ-locHZ-1) + (locTX-locHX-1)) - 20
        a_col = 0
        a_val = ZERO

        ! generate A
        do xx = locHX,locTX
            do zz = locHZ,locTZ

               ! col_tmp and vel_tmp
               col_tmp = 0
               vel_tmp = ZERO
               if (xx > locHX) then
                   if (zz > locHZ) then
                       col_tmp(1) = label(zz-locHZ,  xx-locHX)
                       vel_tmp(1) =   vel(zz-locHZ,  xx-locHX)
                   end if
                       col_tmp(2) = label(zz-locHZ+1,xx-locHX)
                       vel_tmp(2) =   vel(zz-locHZ+1,xx-locHX)
                   if (zz < locTZ) then
                       col_tmp(3) = label(zz-locHZ+2,xx-locHX)
                       vel_tmp(3) =   vel(zz-locHZ+2,xx-locHX)
                   end if
               end if
                   if (zz > locHZ) then
                       col_tmp(4) = label(zz-locHZ,  xx-locHX+1)
                       vel_tmp(4) =   vel(zz-locHZ,  xx-locHX+1)
                   end if                
                       col_tmp(5) = label(zz-locHZ+1,xx-locHX+1)
                       vel_tmp(5) =   vel(zz-locHZ+1,xx-locHX+1)
                   if (zz < locTZ) then               
                       col_tmp(6) = label(zz-locHZ+2,xx-locHX+1)
                       vel_tmp(6) =   vel(zz-locHZ+2,xx-locHX+1)
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       col_tmp(7) = label(zz-locHZ,  xx-locHX+2)
                       vel_tmp(7) =   vel(zz-locHZ,  xx-locHX+2)
                   end if
                       col_tmp(8) = label(zz-locHZ+1,xx-locHX+2)
                       vel_tmp(8) =   vel(zz-locHZ+1,xx-locHX+2)
                   if (zz < locTZ) then
                       col_tmp(9) = label(zz-locHZ+2,xx-locHX+2)
                       vel_tmp(9) =   vel(zz-locHZ+2,xx-locHX+2)
                   end if
               end if

               ! rho_tmp: copy from rho
               rho_tmp = ZERO
               rho_tmp(1) = rho(zz-locHZ+1,xx-locHX+1)
               rho_tmp(2) = rho(zz-locHZ+2,xx-locHX+1)
               rho_tmp(3) = rho(zz-locHZ+3,xx-locHX+1)
               rho_tmp(4) = rho(zz-locHZ+1,xx-locHX+2)
               rho_tmp(5) = rho(zz-locHZ+2,xx-locHX+2)
               rho_tmp(6) = rho(zz-locHZ+3,xx-locHX+2)
               rho_tmp(7) = rho(zz-locHZ+1,xx-locHX+3)
               rho_tmp(8) = rho(zz-locHZ+2,xx-locHX+3)
               rho_tmp(9) = rho(zz-locHZ+3,xx-locHX+3)

               ! value preprocessing
               Czz = -ONE
               Cxx = -ONE
               Czx = ZERO

               ! val_tmp
               val_tmp = ZERO

               ! 1. d^2/dz^2
               tmp  = Czz * rho_tmp(5) / Sz(zz)
               tmp1 = (1.0/(rho_tmp(4)*Sz(zz-1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               tmp2 = (1.0/(rho_tmp(6)*Sz(zz+1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               val_tmp(4) = val_tmp(4) + tmp * tmp1
               val_tmp(5) = val_tmp(5) - tmp *(tmp1 + tmp2)
               val_tmp(6) = val_tmp(6) + tmp * tmp2

               ! 2. d^2/dx^2
               tmp = Cxx * rx**2.0 * rho_tmp(5) / Sx(xx)
               tmp1 = (1.0/(rho_tmp(2)*Sx(xx-1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               tmp2 = (1.0/(rho_tmp(8)*Sx(xx+1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               val_tmp(2) = val_tmp(2) + tmp * tmp1
               val_tmp(5) = val_tmp(5) - tmp *(tmp1 + tmp2)
               val_tmp(8) = val_tmp(8) + tmp * tmp2

               ! 3. d^2/dzdx
               tmp = Czx * rx/4.0 * rho_tmp(5) / (Sz(zz)*Sx(xx))
               val_tmp(1) = val_tmp(1) + tmp * (rho_tmp(2)+rho_tmp(4))/(2.0*rho_tmp(2)*rho_tmp(4))
               val_tmp(3) = val_tmp(3) - tmp * (rho_tmp(2)+rho_tmp(6))/(2.0*rho_tmp(2)*rho_tmp(6))
               val_tmp(7) = val_tmp(7) - tmp * (rho_tmp(8)+rho_tmp(4))/(2.0*rho_tmp(8)*rho_tmp(4))
               val_tmp(9) = val_tmp(9) + tmp * (rho_tmp(8)+rho_tmp(6))/(2.0*rho_tmp(8)*rho_tmp(6))

               ! 4. mass term
               val_tmp(5) = val_tmp(5) - (hz*freq/vel_tmp(5))**2.0

               ! 5. KEY step: modification for the sake of overlapping interface
               if (locHZ > 2 .and. zz == locHZ) then
                   val_tmp(2:8:3) = ZERO
               end if
               if (locHX > 2 .and. xx == locHX) then
                   val_tmp(4:6) = ZERO
               end if

               ! copy val and col
               counter = 0
               lvl = label(zz-locHZ+1,xx-locHX+1)
               do kk = 1,9
                   if (col_tmp(kk) /= 0) then
                       A_col( A_ind(lvl)+counter ) = col_tmp(kk)
                       A_val( A_ind(lvl)+counter ) = val_tmp(kk)
                       counter = counter + 1
                   end if
               end do

            end do
        end do
        deallocate(Sz,Sx)

        return
    end subroutine A2D_acoustic


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Elliptic anisotropy  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A2D_elliptic( freq, Nz, Nx, N_delta, sigma0, hz, hx, locHZ, locTZ, locHX, locTX, label, vel, rho, epslon, theta, A )

        complex(kind=4) :: freq, sigma0, hz, hx
        integer         :: Nz, Nx, N_delta, locHZ, locHX, locTZ, locTX
        integer         ::  label(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    vel(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    rho(locTZ-locHZ+3,locTX-locHX+3)
        real(kind=4)    :: epslon(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::  theta(locTZ-locHZ+1,locTX-locHX+1)
        type(CSR)       :: A

        integer                     :: zz, xx, kk, lvl, length, counter, col_tmp(9)
        complex(kind=4)             :: rx, Czz, Cxx, Czx, tmp, tmp1, tmp2, ee, tt, lambda_z
        complex(kind=4)             :: lambda_x, vel_tmp(9), rho_tmp(9), val_tmp(9)
        complex(kind=4),allocatable :: Sz(:), Sx(:)

        ! the ratio of the spatial step size and parameters
        rx  = hz / hx    

        ! generate Sz and Sx
        allocate(Sz(Nz),Sx(Nx))
        Sz = ONE
        Sx = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function 
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
        end do

        ! allocate A%ind
        length = (locTZ-locHZ+1)*(locTX-locHX+1)    ! the order of the matrix A: number of unknows
        allocate(A%ind(length+1))
        A%ind = 0

        do xx = locHX,locTX
            do zz = locHZ,locTZ
               lvl = label(zz-locHZ+1,xx-locHX+1) + 1
               if (xx > locHX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if                
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then               
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
            end do
        end do

        A%ind(1) = 1
        do kk = 1,length
            A%ind(kk+1) = A%ind(kk+1) + A%ind(kk)
        end do

        ! allocate A%col and A%val
        length = 9*(locTZ-locHZ+1)*(locTX-locHX+1) - 6*((locTZ-locHZ-1) + (locTX-locHX-1)) - 20
        allocate(A%col(length),A%val(length))
        A%col = 0
        A%val = ZERO

        ! generate A
        do xx = locHX,locTX
            do zz = locHZ,locTZ

               ! col_tmp and vel_tmp
               col_tmp = 0
               vel_tmp = ZERO
               if (xx > locHX) then
                   if (zz > locHZ) then
                       col_tmp(1) = label(zz-locHZ,  xx-locHX)
                       vel_tmp(1) =   vel(zz-locHZ,  xx-locHX)
                   end if
                       col_tmp(2) = label(zz-locHZ+1,xx-locHX)
                       vel_tmp(2) =   vel(zz-locHZ+1,xx-locHX)
                   if (zz < locTZ) then
                       col_tmp(3) = label(zz-locHZ+2,xx-locHX)
                       vel_tmp(3) =   vel(zz-locHZ+2,xx-locHX)
                   end if
               end if
                   if (zz > locHZ) then
                       col_tmp(4) = label(zz-locHZ,  xx-locHX+1)
                       vel_tmp(4) =   vel(zz-locHZ,  xx-locHX+1)
                   end if                
                       col_tmp(5) = label(zz-locHZ+1,xx-locHX+1)
                       vel_tmp(5) =   vel(zz-locHZ+1,xx-locHX+1)
                   if (zz < locTZ) then               
                       col_tmp(6) = label(zz-locHZ+2,xx-locHX+1)
                       vel_tmp(6) =   vel(zz-locHZ+2,xx-locHX+1)
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       col_tmp(7) = label(zz-locHZ,  xx-locHX+2)
                       vel_tmp(7) =   vel(zz-locHZ,  xx-locHX+2)
                   end if
                       col_tmp(8) = label(zz-locHZ+1,xx-locHX+2)
                       vel_tmp(8) =   vel(zz-locHZ+1,xx-locHX+2)
                   if (zz < locTZ) then
                       col_tmp(9) = label(zz-locHZ+2,xx-locHX+2)
                       vel_tmp(9) =   vel(zz-locHZ+2,xx-locHX+2)
                   end if
               end if

               ! rho_tmp: copy from rho
               rho_tmp = ZERO
               rho_tmp(1) = rho(zz-locHZ+1,xx-locHX+1)
               rho_tmp(2) = rho(zz-locHZ+2,xx-locHX+1)
               rho_tmp(3) = rho(zz-locHZ+3,xx-locHX+1)
               rho_tmp(4) = rho(zz-locHZ+1,xx-locHX+2)
               rho_tmp(5) = rho(zz-locHZ+2,xx-locHX+2)
               rho_tmp(6) = rho(zz-locHZ+3,xx-locHX+2)
               rho_tmp(7) = rho(zz-locHZ+1,xx-locHX+3)
               rho_tmp(8) = rho(zz-locHZ+2,xx-locHX+3)
               rho_tmp(9) = rho(zz-locHZ+3,xx-locHX+3)

               ! value preprocessing
               ee = epslon(zz-locHZ+1,xx-locHX+1)
               tt =  theta(zz-locHZ+1,xx-locHX+1) * pi/180.0
               lambda_z = -ONE
               lambda_x = -ONE - 2.0*ee
               Czz = lambda_z * cos(tt)**2.0 + lambda_x * sin(tt)**2.0
               Cxx = lambda_z * sin(tt)**2.0 + lambda_x * cos(tt)**2.0
               Czx = (lambda_z - lambda_x) * sin(2.0*tt)

               ! val_tmp
               val_tmp = ZERO

               ! 1. d^2/dz^2
               tmp  = Czz * rho_tmp(5) / Sz(zz)
               tmp1 = (1.0/(rho_tmp(4)*Sz(zz-1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               tmp2 = (1.0/(rho_tmp(6)*Sz(zz+1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               val_tmp(4) = val_tmp(4) + tmp * tmp1
               val_tmp(5) = val_tmp(5) - tmp *(tmp1 + tmp2)
               val_tmp(6) = val_tmp(6) + tmp * tmp2

               ! 2. d^2/dx^2
               tmp = Cxx * rx**2.0 * rho_tmp(5) / Sx(xx)
               tmp1 = (1.0/(rho_tmp(2)*Sx(xx-1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               tmp2 = (1.0/(rho_tmp(8)*Sx(xx+1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               val_tmp(2) = val_tmp(2) + tmp * tmp1
               val_tmp(5) = val_tmp(5) - tmp *(tmp1 + tmp2)
               val_tmp(8) = val_tmp(8) + tmp * tmp2

               ! 3. d^2/dzdx
               tmp = Czx * rx/4.0 * rho_tmp(5) / (Sz(zz)*Sx(xx))
               val_tmp(1) = val_tmp(1) + tmp * (rho_tmp(2)+rho_tmp(4))/(2.0*rho_tmp(2)*rho_tmp(4))
               val_tmp(3) = val_tmp(3) - tmp * (rho_tmp(2)+rho_tmp(6))/(2.0*rho_tmp(2)*rho_tmp(6))
               val_tmp(7) = val_tmp(7) - tmp * (rho_tmp(8)+rho_tmp(4))/(2.0*rho_tmp(8)*rho_tmp(4))
               val_tmp(9) = val_tmp(9) + tmp * (rho_tmp(8)+rho_tmp(6))/(2.0*rho_tmp(8)*rho_tmp(6))

               ! 4. mass term
               val_tmp(5) = val_tmp(5) - (hz*freq/vel_tmp(5))**2.0

               ! 5. KEY step: modification for the sake of overlapping interface
               if (locHZ > 2 .and. zz == locHZ) then
                   val_tmp(2:8:3) = ZERO
               end if
               if (locHX > 2 .and. xx == locHX) then
                   val_tmp(4:6) = ZERO
               end if

               ! copy val and col
               counter = 0
               lvl = label(zz-locHZ+1,xx-locHX+1)
               do kk = 1,9
                   if (col_tmp(kk) /= 0) then
                       A%col( A%ind(lvl)+counter ) = col_tmp(kk)
                       A%val( A%ind(lvl)+counter ) = val_tmp(kk)
                       counter = counter + 1
                   end if
               end do

            end do
        end do
        deallocate(Sz,Sx)

        return
    end subroutine A2D_elliptic


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  VTI  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A2D_VTI( freq, Nz, Nx, N_delta, sigma0, hz, hx, locHZ, locTZ, locHX, locTX, label, vel, rho, epslon, delta, A )

        complex(kind=4) :: freq, sigma0, hz, hx
        integer         :: Nz, Nx, N_delta, locHZ, locHX, locTZ, locTX
        integer         ::  label(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    vel(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    rho(locTZ-locHZ+3,locTX-locHX+3)
        real(kind=4)    :: epslon(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::  delta(locTZ-locHZ+1,locTX-locHX+1)
        type(CSR)       :: A

        integer                     :: zz, xx, kk, lvl, length, counter, col_tmp(9)
        complex(kind=4)             :: rx, Czz, Cxx, Czx, tmp, tmp1, tmp2, tmp3, ee, dd
        complex(kind=4)             :: rhoX(6), rhoZ(2), vel_tmp(9), rho_tmp(9), val_tmp(9)
        complex(kind=4),allocatable :: Sz(:), Sx(:)

        ! the ratio of the spatial step size and parameters
        rx  = hz / hx    

        ! generate Sz and Sx
        allocate(Sz(Nz),Sx(Nx))
        Sz = ONE
        Sx = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function 
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
        end do

        ! allocate A%ind
        length = (locTZ-locHZ+1)*(locTX-locHX+1)    ! the order of the matrix A: number of unknows
        allocate(A%ind(length+1))
        A%ind = 0

        do xx = locHX,locTX
            do zz = locHZ,locTZ
               lvl = label(zz-locHZ+1,xx-locHX+1) + 1
               if (xx > locHX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if                
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then               
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
            end do
        end do

        A%ind(1) = 1
        do kk = 1,length
            A%ind(kk+1) = A%ind(kk+1) + A%ind(kk)
        end do

        ! allocate A%col and A%val
        length = 9*(locTZ-locHZ+1)*(locTX-locHX+1) - 6*((locTZ-locHZ-1) + (locTX-locHX-1)) - 20
        allocate(A%col(length),A%val(length))
        A%col = 0
        A%val = ZERO

        ! generate A
        do xx = locHX,locTX
            do zz = locHZ,locTZ

               ! col_tmp and vel_tmp
               col_tmp = 0
               vel_tmp = ZERO
               if (xx > locHX) then
                   if (zz > locHZ) then
                       col_tmp(1) = label(zz-locHZ,  xx-locHX)
                       vel_tmp(1) =   vel(zz-locHZ,  xx-locHX)
                   end if
                       col_tmp(2) = label(zz-locHZ+1,xx-locHX)
                       vel_tmp(2) =   vel(zz-locHZ+1,xx-locHX)
                   if (zz < locTZ) then
                       col_tmp(3) = label(zz-locHZ+2,xx-locHX)
                       vel_tmp(3) =   vel(zz-locHZ+2,xx-locHX)
                   end if
               end if
                   if (zz > locHZ) then
                       col_tmp(4) = label(zz-locHZ,  xx-locHX+1)
                       vel_tmp(4) =   vel(zz-locHZ,  xx-locHX+1)
                   end if                
                       col_tmp(5) = label(zz-locHZ+1,xx-locHX+1)
                       vel_tmp(5) =   vel(zz-locHZ+1,xx-locHX+1)
                   if (zz < locTZ) then               
                       col_tmp(6) = label(zz-locHZ+2,xx-locHX+1)
                       vel_tmp(6) =   vel(zz-locHZ+2,xx-locHX+1)
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       col_tmp(7) = label(zz-locHZ,  xx-locHX+2)
                       vel_tmp(7) =   vel(zz-locHZ,  xx-locHX+2)
                   end if
                       col_tmp(8) = label(zz-locHZ+1,xx-locHX+2)
                       vel_tmp(8) =   vel(zz-locHZ+1,xx-locHX+2)
                   if (zz < locTZ) then
                       col_tmp(9) = label(zz-locHZ+2,xx-locHX+2)
                       vel_tmp(9) =   vel(zz-locHZ+2,xx-locHX+2)
                   end if
               end if

               ! rho_tmp: copy from rho
               rho_tmp = ZERO
               rho_tmp(1) = rho(zz-locHZ+1,xx-locHX+1)
               rho_tmp(2) = rho(zz-locHZ+2,xx-locHX+1)
               rho_tmp(3) = rho(zz-locHZ+3,xx-locHX+1)
               rho_tmp(4) = rho(zz-locHZ+1,xx-locHX+2)
               rho_tmp(5) = rho(zz-locHZ+2,xx-locHX+2)
               rho_tmp(6) = rho(zz-locHZ+3,xx-locHX+2)
               rho_tmp(7) = rho(zz-locHZ+1,xx-locHX+3)
               rho_tmp(8) = rho(zz-locHZ+2,xx-locHX+3)
               rho_tmp(9) = rho(zz-locHZ+3,xx-locHX+3)

               ! value preprocessing
               ee = epslon(zz-locHZ+1,xx-locHX+1)
               dd =  delta(zz-locHZ+1,xx-locHX+1)
               Czz = -ONE
               Cxx = -ONE - 2.0*ee
               Czx = ZERO

               ! rhoZ and rhoX preprocessing
               rhoX = ZERO
               rhoX(1) = (1.0/(rho_tmp(1)*Sx(xx-1))+1.0/(rho_tmp(4)*Sx(xx))) / 2.0
               rhoX(2) = (1.0/(rho_tmp(7)*Sx(xx+1))+1.0/(rho_tmp(4)*Sx(xx))) / 2.0
               rhoX(3) = (1.0/(rho_tmp(2)*Sx(xx-1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               rhoX(4) = (1.0/(rho_tmp(8)*Sx(xx+1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               rhoX(5) = (1.0/(rho_tmp(3)*Sx(xx-1))+1.0/(rho_tmp(6)*Sx(xx))) / 2.0
               rhoX(6) = (1.0/(rho_tmp(9)*Sx(xx+1))+1.0/(rho_tmp(6)*Sx(xx))) / 2.0
               rhoZ = ZERO
               rhoZ(1) = (1.0/(rho_tmp(4)*Sz(zz-1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               rhoZ(2) = (1.0/(rho_tmp(6)*Sz(zz+1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0

               ! val_tmp
               val_tmp = ZERO

               ! 1. d^2/dz^2
               tmp  = Czz * rho_tmp(5) / Sz(zz)
               val_tmp(4) = val_tmp(4) + tmp * rhoZ(1)
               val_tmp(5) = val_tmp(5) - tmp *(rhoZ(1) + rhoZ(2))
               val_tmp(6) = val_tmp(6) + tmp * rhoZ(2)

               ! 2. d^2/dx^2
               tmp = Cxx * rx**2.0 * rho_tmp(5) / Sx(xx)
               val_tmp(2) = val_tmp(2) + tmp * rhoX(3)
               val_tmp(5) = val_tmp(5) - tmp *(rhoX(3) + rhoX(4))
               val_tmp(8) = val_tmp(8) + tmp * rhoX(4)

               ! 3. d^2/dzdx
               tmp = Czx * rx/4.0 * rho_tmp(5) / (Sz(zz)*Sx(xx))
               val_tmp(1) = val_tmp(1) + tmp * (rho_tmp(2)+rho_tmp(4))/(2.0*rho_tmp(2)*rho_tmp(4))
               val_tmp(3) = val_tmp(3) - tmp * (rho_tmp(2)+rho_tmp(6))/(2.0*rho_tmp(2)*rho_tmp(6))
               val_tmp(7) = val_tmp(7) - tmp * (rho_tmp(8)+rho_tmp(4))/(2.0*rho_tmp(8)*rho_tmp(4))
               val_tmp(9) = val_tmp(9) + tmp * (rho_tmp(8)+rho_tmp(6))/(2.0*rho_tmp(8)*rho_tmp(6))

               ! 4. d^4/dz^2dx^2
               tmp  = -2.0*(ee-dd) * rx**2.0 * rho_tmp(5) / ( (hz*freq/vel_tmp(5))**2.0 * Sz(zz) * Sx(xx) )
               tmp1 = tmp * rho_tmp(4) * rhoZ(1)
               tmp2 = tmp * rho_tmp(5) * (-rhoZ(1)-rhoZ(2))
               tmp3 = tmp * rho_tmp(6) * rhoZ(2)
        
               val_tmp(1) = val_tmp(1) + tmp1 * rhoX(1)
               val_tmp(4) = val_tmp(4) - tmp1 *(rhoX(1)+rhoX(2))
               val_tmp(7) = val_tmp(7) + tmp1 * rhoX(2)
               val_tmp(2) = val_tmp(2) + tmp2 * rhoX(3)
               val_tmp(5) = val_tmp(5) - tmp2 *(rhoX(3)+rhoX(4))
               val_tmp(8) = val_tmp(8) + tmp2 * rhoX(4)
               val_tmp(3) = val_tmp(3) + tmp3 * rhoX(5)
               val_tmp(6) = val_tmp(6) - tmp3 *(rhoX(5)+rhoX(6))
               val_tmp(9) = val_tmp(9) + tmp3 * rhoX(6)

               ! 4. mass term
               val_tmp(5) = val_tmp(5) - (hz*freq/vel_tmp(5))**2.0

               ! 5. KEY step: modification for the sake of overlapping interface
               if (locHZ > 2 .and. zz == locHZ) then
                   val_tmp(2:8:3) = ZERO
               end if
               if (locHX > 2 .and. xx == locHX) then
                   val_tmp(4:6) = ZERO
               end if

               ! copy val and col
               counter = 0
               lvl = label(zz-locHZ+1,xx-locHX+1)
               do kk = 1,9
                   if (col_tmp(kk) /= 0) then
                       A%col( A%ind(lvl)+counter ) = col_tmp(kk)
                       A%val( A%ind(lvl)+counter ) = val_tmp(kk)
                       counter = counter + 1
                   end if
               end do

            end do
        end do
        deallocate(Sz,Sx)

        return
    end subroutine A2D_VTI


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  HTI  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A2D_HTI( freq, Nz, Nx, N_delta, sigma0, hz, hx, locHZ, locTZ, locHX, locTX, label, vel, rho, epslon, delta, A )

        complex(kind=4) :: freq, sigma0, hz, hx
        integer         :: Nz, Nx, N_delta, locHZ, locHX, locTZ, locTX
        integer         ::  label(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    vel(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::    rho(locTZ-locHZ+3,locTX-locHX+3)
        real(kind=4)    :: epslon(locTZ-locHZ+1,locTX-locHX+1)
        real(kind=4)    ::  delta(locTZ-locHZ+1,locTX-locHX+1)
        type(CSR)       :: A

        integer                     :: zz, xx, kk, lvl, length, counter, col_tmp(9)
        complex(kind=4)             :: rx, Czz, Cxx, Czx, tmp, tmp1, tmp2, tmp3, ee, dd
        complex(kind=4)             :: rhoX(2), rhoZ(6), vel_tmp(9), rho_tmp(9), val_tmp(9)
        complex(kind=4),allocatable :: Sz(:), Sx(:)

        ! the ratio of the spatial step size and parameters
        rx  = hz / hx    

        ! generate Sz and Sx
        allocate(Sz(Nz),Sx(Nx))
        Sz = ONE
        Sx = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function 
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
        end do

        ! allocate A%ind
        length = (locTZ-locHZ+1)*(locTX-locHX+1)    ! the order of the matrix A: number of unknows
        allocate(A%ind(length+1))
        A%ind = 0

        do xx = locHX,locTX
            do zz = locHZ,locTZ
               lvl = label(zz-locHZ+1,xx-locHX+1) + 1
               if (xx > locHX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if                
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then               
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
                       A%ind(lvl) = A%ind(lvl) + 1
                   if (zz < locTZ) then
                       A%ind(lvl) = A%ind(lvl) + 1
                   end if
               end if
            end do
        end do

        A%ind(1) = 1
        do kk = 1,length
            A%ind(kk+1) = A%ind(kk+1) + A%ind(kk)
        end do

        ! allocate A%col and A%val
        length = 9*(locTZ-locHZ+1)*(locTX-locHX+1) - 6*((locTZ-locHZ-1) + (locTX-locHX-1)) - 20
        allocate(A%col(length),A%val(length))
        A%col = 0
        A%val = ZERO

        ! generate A
        do xx = locHX,locTX
            do zz = locHZ,locTZ

               ! col_tmp and vel_tmp
               col_tmp = 0
               vel_tmp = ZERO
               if (xx > locHX) then
                   if (zz > locHZ) then
                       col_tmp(1) = label(zz-locHZ,  xx-locHX)
                       vel_tmp(1) =   vel(zz-locHZ,  xx-locHX)
                   end if
                       col_tmp(2) = label(zz-locHZ+1,xx-locHX)
                       vel_tmp(2) =   vel(zz-locHZ+1,xx-locHX)
                   if (zz < locTZ) then
                       col_tmp(3) = label(zz-locHZ+2,xx-locHX)
                       vel_tmp(3) =   vel(zz-locHZ+2,xx-locHX)
                   end if
               end if
                   if (zz > locHZ) then
                       col_tmp(4) = label(zz-locHZ,  xx-locHX+1)
                       vel_tmp(4) =   vel(zz-locHZ,  xx-locHX+1)
                   end if                
                       col_tmp(5) = label(zz-locHZ+1,xx-locHX+1)
                       vel_tmp(5) =   vel(zz-locHZ+1,xx-locHX+1)
                   if (zz < locTZ) then               
                       col_tmp(6) = label(zz-locHZ+2,xx-locHX+1)
                       vel_tmp(6) =   vel(zz-locHZ+2,xx-locHX+1)
                   end if
               if (xx < locTX) then
                   if (zz > locHZ) then
                       col_tmp(7) = label(zz-locHZ,  xx-locHX+2)
                       vel_tmp(7) =   vel(zz-locHZ,  xx-locHX+2)
                   end if
                       col_tmp(8) = label(zz-locHZ+1,xx-locHX+2)
                       vel_tmp(8) =   vel(zz-locHZ+1,xx-locHX+2)
                   if (zz < locTZ) then
                       col_tmp(9) = label(zz-locHZ+2,xx-locHX+2)
                       vel_tmp(9) =   vel(zz-locHZ+2,xx-locHX+2)
                   end if
               end if

               ! rho_tmp: copy from rho
               rho_tmp = ZERO
               rho_tmp(1) = rho(zz-locHZ+1,xx-locHX+1)
               rho_tmp(2) = rho(zz-locHZ+2,xx-locHX+1)
               rho_tmp(3) = rho(zz-locHZ+3,xx-locHX+1)
               rho_tmp(4) = rho(zz-locHZ+1,xx-locHX+2)
               rho_tmp(5) = rho(zz-locHZ+2,xx-locHX+2)
               rho_tmp(6) = rho(zz-locHZ+3,xx-locHX+2)
               rho_tmp(7) = rho(zz-locHZ+1,xx-locHX+3)
               rho_tmp(8) = rho(zz-locHZ+2,xx-locHX+3)
               rho_tmp(9) = rho(zz-locHZ+3,xx-locHX+3)

               ! value preprocessing
               ee = epslon(zz-locHZ+1,xx-locHX+1)
               dd =  delta(zz-locHZ+1,xx-locHX+1)
               Czz = -ONE - 2.0*ee
               Cxx = -ONE
               Czx = ZERO

               ! rhoZ and rhoX preprocessing
               rhoZ = ZERO
               rhoZ(1) = (1.0/(rho_tmp(1)*Sz(zz-1))+1.0/(rho_tmp(2)*Sz(zz))) / 2.0
               rhoZ(2) = (1.0/(rho_tmp(3)*Sz(zz+1))+1.0/(rho_tmp(2)*Sz(zz))) / 2.0
               rhoZ(3) = (1.0/(rho_tmp(4)*Sz(zz-1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               rhoZ(4) = (1.0/(rho_tmp(6)*Sz(zz+1))+1.0/(rho_tmp(5)*Sz(zz))) / 2.0
               rhoZ(5) = (1.0/(rho_tmp(7)*Sz(zz-1))+1.0/(rho_tmp(8)*Sz(zz))) / 2.0
               rhoZ(6) = (1.0/(rho_tmp(9)*Sz(zz+1))+1.0/(rho_tmp(8)*Sz(zz))) / 2.0
               rhoX = ZERO
               rhoX(1) = (1.0/(rho_tmp(2)*Sx(xx-1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0
               rhoX(2) = (1.0/(rho_tmp(8)*Sx(xx+1))+1.0/(rho_tmp(5)*Sx(xx))) / 2.0

               ! val_tmp
               val_tmp = ZERO

               ! 1. d^2/dz^2
               tmp  = Czz * rho_tmp(5) / Sz(zz)
               val_tmp(4) = val_tmp(4) + tmp * rhoZ(3)
               val_tmp(5) = val_tmp(5) - tmp *(rhoZ(3) + rhoZ(4))
               val_tmp(6) = val_tmp(6) + tmp * rhoZ(4)

               ! 2. d^2/dx^2
               tmp = Cxx * rx**2.0 * rho_tmp(5) / Sx(xx)
               val_tmp(2) = val_tmp(2) + tmp * rhoX(1)
               val_tmp(5) = val_tmp(5) - tmp *(rhoX(1) + rhoX(2))
               val_tmp(8) = val_tmp(8) + tmp * rhoX(2)

               ! 3. d^2/dzdx
               tmp = Czx * rx/4.0 * rho_tmp(5) / (Sz(zz)*Sx(xx))
               val_tmp(1) = val_tmp(1) + tmp * (rho_tmp(2)+rho_tmp(4))/(2.0*rho_tmp(2)*rho_tmp(4))
               val_tmp(3) = val_tmp(3) - tmp * (rho_tmp(2)+rho_tmp(6))/(2.0*rho_tmp(2)*rho_tmp(6))
               val_tmp(7) = val_tmp(7) - tmp * (rho_tmp(8)+rho_tmp(4))/(2.0*rho_tmp(8)*rho_tmp(4))
               val_tmp(9) = val_tmp(9) + tmp * (rho_tmp(8)+rho_tmp(6))/(2.0*rho_tmp(8)*rho_tmp(6))

               ! 4. d^4/dz^2dx^2
               tmp  = -2.0*(ee-dd) * rx**2.0 * rho_tmp(5) / ( (hz*freq/vel_tmp(5))**2.0 * Sz(zz) * Sx(xx) )
               tmp1 = tmp * rho_tmp(2) * rhoX(1)
               tmp2 = tmp * rho_tmp(5) * (-rhoX(1)-rhoX(2))
               tmp3 = tmp * rho_tmp(8) * rhoX(2)

               val_tmp(1) = val_tmp(1) + tmp1 * rhoZ(1)
               val_tmp(2) = val_tmp(2) - tmp1 *(rhoZ(1)+rhoZ(2))
               val_tmp(3) = val_tmp(3) + tmp1 * rhoZ(2)
               val_tmp(4) = val_tmp(4) + tmp2 * rhoZ(3)
               val_tmp(5) = val_tmp(5) - tmp2 *(rhoZ(3)+rhoZ(4))
               val_tmp(6) = val_tmp(6) + tmp2 * rhoZ(4)
               val_tmp(7) = val_tmp(7) + tmp3 * rhoZ(5)
               val_tmp(8) = val_tmp(8) - tmp3 *(rhoZ(5)+rhoZ(6))
               val_tmp(9) = val_tmp(9) + tmp3 * rhoZ(6)        

               ! 4. mass term
               val_tmp(5) = val_tmp(5) - (hz*freq/vel_tmp(5))**2.0


               ! 5. KEY step: modification for the sake of overlapping interface
               if (locHZ > 2 .and. zz == locHZ) then
                   val_tmp(2:8:3) = ZERO
               end if
               if (locHX > 2 .and. xx == locHX) then
                   val_tmp(4:6) = ZERO
               end if

               ! copy val and col
               counter = 0
               lvl = label(zz-locHZ+1,xx-locHX+1)
               do kk = 1,9
                   if (col_tmp(kk) /= 0) then
                       A%col( A%ind(lvl)+counter ) = col_tmp(kk)
                       A%val( A%ind(lvl)+counter ) = val_tmp(kk)
                       counter = counter + 1
                   end if
               end do

            end do
        end do
        deallocate(Sz,Sx)

        return
    end subroutine A2D_HTI


end module A2D
