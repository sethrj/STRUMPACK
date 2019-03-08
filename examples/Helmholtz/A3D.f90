module A3D

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  3D Helmholtz matrix generation  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!use NDpre3D
use partition
implicit none
contains


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  acoustic case  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A3D_acoustic( freq, Nx, Ny, Nz, N_delta, sigma0, hx, hy, hz, locHX, locTX, &
         locHY, locTY, locHZ, locTZ, label, vel, rho, n, nnz, a_ind, a_col, a_val)

        complex(kind=4) :: freq, sigma0, hx, hy, hz
        integer         :: Nx, Ny, Nz, N_delta, locHX, locTX, locHY, locTY, locHZ, locTZ
        integer         :: label(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::   vel(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::   rho(locTX-locHX+3,locTY-locHY+3,locTZ-locHZ+3)
        integer :: n, nnz, a_ind(n), a_col(nnz)
        complex(kind=4) :: a_val(nnz)
!        type(CSR)       :: A

        integer                     :: xx, yy, zz, kk, lvl, length, counter, col_tmp(27)
        complex(kind=4)             :: rx, ry, Cxx, Cyy, Czz, Cxy, Cyz, Czx, tmp, tmp1
        complex(kind=4)             :: tmp2, vel_tmp(27), rho_tmp(27), val_tmp(27)
        complex(kind=4),allocatable :: Sx(:), Sy(:), Sz(:)

        ! the parameters
        rx  = hz / hx
        ry  = hz / hy

        ! generating Sx, Sy and Sz
        allocate(Sx(Nx),Sy(Ny),Sz(Nz))
        Sx = ONE
        Sy = ONE
        Sz = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
            Sy(counter)      = Sy(counter) - tmp
            Sy(Ny-counter+1) = Sy(counter)
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)
        end do

        length = (locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1)    ! the order of A
!        allocate(a_ind(length+1))
        a_ind = 0

        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1) + 1
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                    end if

                end do
            end do
        end do

        a_ind(1) = 1
        do kk = 1,length
            a_ind(kk+1) = a_ind(kk+1) + a_ind(kk)
        end do

        ! allocating A%col and A%val
        length = 27*(locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1) &
             - 18*((locTX-locHX-1)*(locTY-locHY-1)+(locTY-locHY-1)*(locTZ-locHZ-1) &
             +(locTZ-locHZ-1)*(locTX-locHX-1)) - 60*((locTX-locHX-1)+(locTY-locHY-1)&
             +(locTZ-locHZ-1)) - 19*8
!        allocate(a_col(length),a_val(length))
        a_col = 0
        a_val = ZERO

        ! generating A
        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    ! col_tmp and vel_tmp
                    col_tmp = 0
                    vel_tmp = ZERO
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(1) = label(xx-locHX,   yy-locHY,   zz-locHZ)
                                vel_tmp(1) =   vel(xx-locHX,   yy-locHY,   zz-locHZ)
                            end if
                                col_tmp(2) = label(xx-locHX+1, yy-locHY,   zz-locHZ)
                                vel_tmp(2) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(3) = label(xx-locHX+2, yy-locHY,   zz-locHZ)
                                vel_tmp(3) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(4) = label(xx-locHX,   yy-locHY+1, zz-locHZ)
                                vel_tmp(4) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ)
                            end if
                                col_tmp(5) = label(xx-locHX+1, yy-locHY+1, zz-locHZ)
                                vel_tmp(5) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(6) = label(xx-locHX+2, yy-locHY+1, zz-locHZ)
                                vel_tmp(6) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(7) = label(xx-locHX,   yy-locHY+2, zz-locHZ)
                                vel_tmp(7) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ)
                            end if
                                col_tmp(8) = label(xx-locHX+1, yy-locHY+2, zz-locHZ)
                                vel_tmp(8) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(9) = label(xx-locHX+2, yy-locHY+2, zz-locHZ)
                                vel_tmp(9) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ)
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(10) = label(xx-locHX,   yy-locHY,   zz-locHZ+1)
                                vel_tmp(10) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+1)
                            end if
                                col_tmp(11) = label(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                                vel_tmp(11) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(12) = label(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                                vel_tmp(12) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(13) = label(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                                vel_tmp(13) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                            end if
                                col_tmp(14) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(14) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(15) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(15) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(16) = label(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                                vel_tmp(16) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                            end if
                                col_tmp(17) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(17) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(18) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(18) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(19) = label(xx-locHX,   yy-locHY,   zz-locHZ+2)
                                vel_tmp(19) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+2)
                            end if
                                col_tmp(20) = label(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                                vel_tmp(20) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(21) = label(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                                vel_tmp(21) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(22) = label(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                                vel_tmp(22) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                            end if
                                col_tmp(23) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(23) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(24) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(24) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(25) = label(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                                vel_tmp(25) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                            end if
                                col_tmp(26) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(26) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(27) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(27) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                            end if
                        end if
                    end if

                    ! rho_tmp
                    rho_tmp = ZERO
                    rho_tmp(1)  = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(2)  = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(3)  = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(4)  = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(5)  = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(6)  = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(7)  = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(8)  = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(9)  = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(10) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(11) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(12) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(13) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(14) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(15) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(16) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(17) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(18) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(19) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(20) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(21) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(22) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(23) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(24) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(25) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(26) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(27) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+3)


                    ! value preprocessing
                    Cxx = -ONE
                    Cyy = -ONE
                    Czz = -ONE
                    Cxy = ZERO
                    Cyz = ZERO
                    Czx = ZERO

                    ! val_tmp
                    val_tmp = ZERO

                    ! 1. d^2/dx^2
                    tmp  = Cxx * rx**2.0 * rho_tmp(14) / Sx(xx)
                    tmp1 = (1.0/(rho_tmp(13)*Sx(xx-1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    tmp2 = (1.0/(rho_tmp(15)*Sx(xx+1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    val_tmp(13) = val_tmp(13) + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(15) = val_tmp(15) + tmp * tmp2
            
                    ! 2. d^2/dy^2
                    tmp  = Cyy * ry**2.0 * rho_tmp(14) / Sy(yy)
                    tmp1 = (1.0/(rho_tmp(11)*Sy(yy-1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    tmp2 = (1.0/(rho_tmp(17)*Sy(yy+1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    val_tmp(11) = val_tmp(11) + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(17) = val_tmp(17) + tmp * tmp2
            
                    ! 3. d^2/dz^2
                    tmp  = Czz * rho_tmp(14) / Sz(zz)
                    tmp1 = (1.0/(rho_tmp(5) *Sz(zz-1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    tmp2 = (1.0/(rho_tmp(23)*Sz(zz+1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    val_tmp(5)  = val_tmp(5)  + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(23) = val_tmp(23) + tmp * tmp2
            
                    ! 4. d^2/dxdy
                    tmp = Cxy * rx*ry/4.0 * rho_tmp(14) / (Sx(xx)*Sy(yy))
                    val_tmp(10) = val_tmp(10) + tmp * (rho_tmp(11)+rho_tmp(13))/(2.0*rho_tmp(11)*rho_tmp(13))
                    val_tmp(12) = val_tmp(12) - tmp * (rho_tmp(11)+rho_tmp(15))/(2.0*rho_tmp(11)*rho_tmp(15))
                    val_tmp(16) = val_tmp(16) - tmp * (rho_tmp(17)+rho_tmp(13))/(2.0*rho_tmp(17)*rho_tmp(13))
                    val_tmp(18) = val_tmp(18) + tmp * (rho_tmp(17)+rho_tmp(15))/(2.0*rho_tmp(17)*rho_tmp(15))
            
                    ! 5. d^2/dydz
                    tmp = Cyz * ry/4.0 * rho_tmp(14) / (Sy(yy)*Sz(zz))
                    val_tmp(2)  = val_tmp(2)  + tmp * (rho_tmp(5) +rho_tmp(11))/(2.0*rho_tmp(5) *rho_tmp(11))
                    val_tmp(8)  = val_tmp(8)  - tmp * (rho_tmp(5) +rho_tmp(17))/(2.0*rho_tmp(5) *rho_tmp(17))
                    val_tmp(20) = val_tmp(20) - tmp * (rho_tmp(23)+rho_tmp(11))/(2.0*rho_tmp(23)*rho_tmp(11))
                    val_tmp(26) = val_tmp(26) + tmp * (rho_tmp(23)+rho_tmp(17))/(2.0*rho_tmp(23)*rho_tmp(17))
            
                    ! 6. d^2/dzdx
                    tmp = Czx * rx/4.0 * rho_tmp(14) / (Sz(zz)*Sx(xx))
                    val_tmp(4)  = val_tmp(4)  + tmp * (rho_tmp(5) +rho_tmp(13))/(2.0*rho_tmp(5) *rho_tmp(13))
                    val_tmp(6)  = val_tmp(6)  - tmp * (rho_tmp(5) +rho_tmp(15))/(2.0*rho_tmp(5) *rho_tmp(15))
                    val_tmp(22) = val_tmp(22) - tmp * (rho_tmp(23)+rho_tmp(13))/(2.0*rho_tmp(23)*rho_tmp(13))
                    val_tmp(24) = val_tmp(24) + tmp * (rho_tmp(23)+rho_tmp(15))/(2.0*rho_tmp(23)*rho_tmp(15))
            
                    ! 7. mass term
                    val_tmp(14) = val_tmp(14) - (hz*freq/vel_tmp(14))**2.0

                    ! 8. KEY step: modification for the sake of overlapping interface
                    if (locHX > 2 .and. xx == locHX) then
                        val_tmp(2:26:3) = ZERO
                    end if
                    if (locHY > 2 .and. yy == locHY) then
                        val_tmp(4:6)   = ZERO
                        val_tmp(13:15) = ZERO
                        val_tmp(22:24) = ZERO                       
                    end if
                    if (locHZ > 2 .and. zz == locHZ) then
                        val_tmp(10:18) = ZERO                        
                    end if
                   
                    ! copy val and col
                    counter = 0
                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    do kk = 1,27
                        if (col_tmp(kk) /= 0) then
                            a_col( a_ind(lvl)+counter ) = col_tmp(kk)
                            a_val( a_ind(lvl)+counter ) = val_tmp(kk)
                            counter = counter + 1
                        end if
                    end do

                end do
            end do
        end do
        deallocate(Sx,Sy,Sz)

        return
    end subroutine A3D_acoustic



    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  elliptic anisotropy  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A3D_elliptic( freq, Nx, Ny, Nz, N_delta, sigma0, hx, hy, hz, locHX, locTX, &
         locHY, locTY, locHZ, locTZ, label, vel, rho, epslon, theta, phi, A )

        complex(kind=4) :: freq, sigma0, hx, hy, hz
        integer         :: Nx, Ny, Nz, N_delta, locHX, locTX, locHY, locTY, locHZ, locTZ
        integer         ::  label(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    vel(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    rho(locTX-locHX+3,locTY-locHY+3,locTZ-locHZ+3)
        real(kind=4)    :: epslon(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::  theta(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    phi(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        type(CSR)       :: A

        integer                     :: xx, yy, zz, kk, lvl, length, counter, col_tmp(27)
        complex(kind=4)             :: rx, ry, Cxx, Cyy, Czz, Cxy, Cyz, Czx, tmp, tmp1 
        complex(kind=4)             :: tmp2, ee, tt, pp, lambda_z, lambda_x, vel_tmp(27) 
        complex(kind=4)             :: rho_tmp(27), val_tmp(27)
        complex(kind=4),allocatable :: Sx(:), Sy(:), Sz(:)

        ! the parameters
        rx  = hz / hx
        ry  = hz / hy

        ! generating Sx, Sy and Sz
        allocate(Sx(Nx),Sy(Ny),Sz(Nz))
        Sx = ONE
        Sy = ONE
        Sz = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
            Sy(counter)      = Sy(counter) - tmp
            Sy(Ny-counter+1) = Sy(counter)
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)      
        end do

        ! allocating A%ind
        length = (locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1)    ! the order of A
        allocate(A%ind(length+1))
        A%ind = 0

        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1) + 1
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                    end if

                end do
            end do
        end do

        A%ind(1) = 1
        do kk = 1,length
            A%ind(kk+1) = A%ind(kk+1) + A%ind(kk)
        end do

        ! allocating A%col and A%val
        length = 27*(locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1) &
             - 18*((locTX-locHX-1)*(locTY-locHY-1)+(locTY-locHY-1)*(locTZ-locHZ-1)&
             +(locTZ-locHZ-1)*(locTX-locHX-1)) - 60*((locTX-locHX-1)+(locTY-locHY-1)&
             +(locTZ-locHZ-1)) - 19*8
        allocate(A%col(length),A%val(length))
        A%col = 0
        A%val = ZERO

        ! generating A
        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    ! col_tmp and vel_tmp
                    col_tmp = 0
                    vel_tmp = ZERO
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(1) = label(xx-locHX,   yy-locHY,   zz-locHZ)
                                vel_tmp(1) =   vel(xx-locHX,   yy-locHY,   zz-locHZ)
                            end if
                                col_tmp(2) = label(xx-locHX+1, yy-locHY,   zz-locHZ)
                                vel_tmp(2) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(3) = label(xx-locHX+2, yy-locHY,   zz-locHZ)
                                vel_tmp(3) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(4) = label(xx-locHX,   yy-locHY+1, zz-locHZ)
                                vel_tmp(4) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ)
                            end if
                                col_tmp(5) = label(xx-locHX+1, yy-locHY+1, zz-locHZ)
                                vel_tmp(5) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(6) = label(xx-locHX+2, yy-locHY+1, zz-locHZ)
                                vel_tmp(6) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(7) = label(xx-locHX,   yy-locHY+2, zz-locHZ)
                                vel_tmp(7) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ)
                            end if
                                col_tmp(8) = label(xx-locHX+1, yy-locHY+2, zz-locHZ)
                                vel_tmp(8) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(9) = label(xx-locHX+2, yy-locHY+2, zz-locHZ)
                                vel_tmp(9) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ)
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(10) = label(xx-locHX,   yy-locHY,   zz-locHZ+1)
                                vel_tmp(10) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+1)
                            end if
                                col_tmp(11) = label(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                                vel_tmp(11) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(12) = label(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                                vel_tmp(12) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(13) = label(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                                vel_tmp(13) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                            end if
                                col_tmp(14) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(14) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(15) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(15) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(16) = label(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                                vel_tmp(16) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                            end if
                                col_tmp(17) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(17) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(18) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(18) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(19) = label(xx-locHX,   yy-locHY,   zz-locHZ+2)
                                vel_tmp(19) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+2)
                            end if
                                col_tmp(20) = label(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                                vel_tmp(20) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(21) = label(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                                vel_tmp(21) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(22) = label(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                                vel_tmp(22) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                            end if
                                col_tmp(23) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(23) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(24) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(24) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(25) = label(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                                vel_tmp(25) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                            end if
                                col_tmp(26) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(26) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(27) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(27) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                            end if
                        end if
                    end if

                    ! rho_tmp
                    rho_tmp = ZERO
                    rho_tmp(1)  = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(2)  = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(3)  = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(4)  = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(5)  = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(6)  = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(7)  = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(8)  = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(9)  = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(10) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(11) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(12) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(13) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(14) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(15) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(16) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(17) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(18) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(19) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(20) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(21) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(22) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(23) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(24) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(25) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(26) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(27) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+3)
                    
                    ! value preprocessing
                    ee = epslon(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    tt =  theta(xx-locHX+1,yy-locHY+1,zz-locHZ+1) * pi/180.0
                    pp =    phi(xx-locHX+1,yy-locHY+1,zz-locHZ+1) * pi/180.0
                    lambda_z = -ONE
                    lambda_x = -ONE - 2.0*ee
                    Cxx = lambda_z * sin(tt)**2.0 * cos(pp)**2.0 + lambda_x * ( cos(tt)**2.0*cos(pp)**2.0 + sin(pp)**2.0 )
                    Cyy = lambda_z * sin(tt)**2.0 * sin(pp)**2.0 + lambda_x * ( cos(tt)**2.0*sin(pp)**2.0 + cos(pp)**2.0 )
                    Czz = lambda_z * cos(tt)**2.0 + lambda_x * sin(tt)**2.0
                    Cxy = (lambda_z - lambda_x) * sin(tt)**2.0 * sin(2.0*pp)
                    Cyz = (lambda_z - lambda_x) * sin(2.0*tt) * sin(pp)
                    Czx = (lambda_z - lambda_x) * sin(2.0*tt) * cos(pp)

                    ! val_tmp
                    val_tmp = ZERO

                    ! 1. d^2/dx^2
                    tmp  = Cxx * rx**2.0 * rho_tmp(14) / Sx(xx)
                    tmp1 = (1.0/(rho_tmp(13)*Sx(xx-1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    tmp2 = (1.0/(rho_tmp(15)*Sx(xx+1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    val_tmp(13) = val_tmp(13) + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(15) = val_tmp(15) + tmp * tmp2
            
                    ! 2. d^2/dy^2
                    tmp  = Cyy * ry**2.0 * rho_tmp(14) / Sy(yy)
                    tmp1 = (1.0/(rho_tmp(11)*Sy(yy-1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    tmp2 = (1.0/(rho_tmp(17)*Sy(yy+1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    val_tmp(11) = val_tmp(11) + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(17) = val_tmp(17) + tmp * tmp2
            
                    ! 3. d^2/dz^2
                    tmp  = Czz * rho_tmp(14) / Sz(zz)
                    tmp1 = (1.0/(rho_tmp(5) *Sz(zz-1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    tmp2 = (1.0/(rho_tmp(23)*Sz(zz+1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    val_tmp(5)  = val_tmp(5)  + tmp * tmp1
                    val_tmp(14) = val_tmp(14) - tmp *(tmp1 + tmp2)
                    val_tmp(23) = val_tmp(23) + tmp * tmp2
            
                    ! 4. d^2/dxdy
                    tmp = Cxy * rx*ry/4.0 * rho_tmp(14) / (Sx(xx)*Sy(yy))
                    val_tmp(10) = val_tmp(10) + tmp * (rho_tmp(11)+rho_tmp(13))/(2.0*rho_tmp(11)*rho_tmp(13))
                    val_tmp(12) = val_tmp(12) - tmp * (rho_tmp(11)+rho_tmp(15))/(2.0*rho_tmp(11)*rho_tmp(15))
                    val_tmp(16) = val_tmp(16) - tmp * (rho_tmp(17)+rho_tmp(13))/(2.0*rho_tmp(17)*rho_tmp(13))
                    val_tmp(18) = val_tmp(18) + tmp * (rho_tmp(17)+rho_tmp(15))/(2.0*rho_tmp(17)*rho_tmp(15))
            
                    ! 5. d^2/dydz
                    tmp = Cyz * ry/4.0 * rho_tmp(14) / (Sy(yy)*Sz(zz))
                    val_tmp(2)  = val_tmp(2)  + tmp * (rho_tmp(5) +rho_tmp(11))/(2.0*rho_tmp(5) *rho_tmp(11))
                    val_tmp(8)  = val_tmp(8)  - tmp * (rho_tmp(5) +rho_tmp(17))/(2.0*rho_tmp(5) *rho_tmp(17))
                    val_tmp(20) = val_tmp(20) - tmp * (rho_tmp(23)+rho_tmp(11))/(2.0*rho_tmp(23)*rho_tmp(11))
                    val_tmp(26) = val_tmp(26) + tmp * (rho_tmp(23)+rho_tmp(17))/(2.0*rho_tmp(23)*rho_tmp(17))
            
                    ! 6. d^2/dzdx
                    tmp = Czx * rx/4.0 * rho_tmp(14) / (Sz(zz)*Sx(xx))
                    val_tmp(4)  = val_tmp(4)  + tmp * (rho_tmp(5) +rho_tmp(13))/(2.0*rho_tmp(5) *rho_tmp(13))
                    val_tmp(6)  = val_tmp(6)  - tmp * (rho_tmp(5) +rho_tmp(15))/(2.0*rho_tmp(5) *rho_tmp(15))
                    val_tmp(22) = val_tmp(22) - tmp * (rho_tmp(23)+rho_tmp(13))/(2.0*rho_tmp(23)*rho_tmp(13))
                    val_tmp(24) = val_tmp(24) + tmp * (rho_tmp(23)+rho_tmp(15))/(2.0*rho_tmp(23)*rho_tmp(15))
            
                    ! 7. mass term
                    val_tmp(14) = val_tmp(14) - (hz*freq/vel_tmp(14))**2.0


                    ! 8. KEY step: modification for the sake of overlapping interface
                    if (locHX > 2 .and. xx == locHX) then
                        val_tmp(2:26:3) = ZERO
                    end if
                    if (locHY > 2 .and. yy == locHY) then
                        val_tmp(4:6)   = ZERO
                        val_tmp(13:15) = ZERO
                        val_tmp(22:24) = ZERO                       
                    end if
                    if (locHZ > 2 .and. zz == locHZ) then
                        val_tmp(10:18) = ZERO                        
                    end if
                   
                    ! copy val and col
                    counter = 0
                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    do kk = 1,27
                        if (col_tmp(kk) /= 0) then
                            A%col( A%ind(lvl)+counter ) = col_tmp(kk)
                            A%val( A%ind(lvl)+counter ) = val_tmp(kk)
                            counter = counter + 1
                        end if
                    end do

                end do
            end do
        end do
        deallocate(Sx,Sy,Sz)

        return
    end subroutine A3D_elliptic


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  VTI  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A3D_VTI( freq, Nx, Ny, Nz, N_delta, sigma0, hx, hy, hz, locHX, locTX, &
         locHY, locTY, locHZ, locTZ, label, vel, rho, epslon, delta, A )

        complex(kind=4) :: freq, sigma0, hx, hy, hz
        integer         :: Nx, Ny, Nz, N_delta, locHX, locTX, locHY, locTY, locHZ, locTZ
        integer         ::  label(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    vel(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    rho(locTX-locHX+3,locTY-locHY+3,locTZ-locHZ+3)
        real(kind=4)    :: epslon(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::  delta(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        type(CSR)       :: A

        integer                     :: xx, yy, zz, kk, lvl, length, counter, col_tmp(27)
        complex(kind=4)             :: rx, ry, Cxx, Cyy, Czz, Cxy, Cyz, Czx, ee, dd
        complex(kind=4)             :: tmp, tmp1, tmp2, tmp3
        complex(kind=4)             :: rhoX(6), rhoY(6), rhoZ(2), vel_tmp(27)
        complex(kind=4)             :: rho_tmp(27), val_tmp(27)
        complex(kind=4),allocatable :: Sx(:), Sy(:), Sz(:)

        ! the parameters
        rx  = hz / hx
        ry  = hz / hy

        ! generating Sx, Sy and Sz
        allocate(Sx(Nx),Sy(Ny),Sz(Nz))
        Sx = ONE
        Sy = ONE
        Sz = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
            Sy(counter)      = Sy(counter) - tmp
            Sy(Ny-counter+1) = Sy(counter)
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)      
        end do

        ! allocating A%ind
        length = (locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1)    ! the order of A
        allocate(A%ind(length+1))
        A%ind = 0

        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1) + 1
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                                A%ind(lvl) = A%ind(lvl) + 1
                            if (xx < locTX) then
                                A%ind(lvl) = A%ind(lvl) + 1
                            end if
                        end if
                    end if

                end do
            end do
        end do

        A%ind(1) = 1
        do kk = 1,length
            A%ind(kk+1) = A%ind(kk+1) + A%ind(kk)
        end do

        ! allocating A%col and A%val
        length = 27*(locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1) &
             - 18*((locTX-locHX-1)*(locTY-locHY-1)+(locTY-locHY-1)*(locTZ-locHZ-1)&
             +(locTZ-locHZ-1)*(locTX-locHX-1)) - 60*((locTX-locHX-1)+(locTY-locHY-1)&
             +(locTZ-locHZ-1)) - 19*8
        allocate(A%col(length),A%val(length))
        A%col = 0
        A%val = ZERO

        ! generating A
        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    ! col_tmp and vel_tmp
                    col_tmp = 0
                    vel_tmp = ZERO
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(1) = label(xx-locHX,   yy-locHY,   zz-locHZ)
                                vel_tmp(1) =   vel(xx-locHX,   yy-locHY,   zz-locHZ)
                            end if
                                col_tmp(2) = label(xx-locHX+1, yy-locHY,   zz-locHZ)
                                vel_tmp(2) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(3) = label(xx-locHX+2, yy-locHY,   zz-locHZ)
                                vel_tmp(3) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(4) = label(xx-locHX,   yy-locHY+1, zz-locHZ)
                                vel_tmp(4) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ)
                            end if
                                col_tmp(5) = label(xx-locHX+1, yy-locHY+1, zz-locHZ)
                                vel_tmp(5) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(6) = label(xx-locHX+2, yy-locHY+1, zz-locHZ)
                                vel_tmp(6) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(7) = label(xx-locHX,   yy-locHY+2, zz-locHZ)
                                vel_tmp(7) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ)
                            end if
                                col_tmp(8) = label(xx-locHX+1, yy-locHY+2, zz-locHZ)
                                vel_tmp(8) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(9) = label(xx-locHX+2, yy-locHY+2, zz-locHZ)
                                vel_tmp(9) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ)
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(10) = label(xx-locHX,   yy-locHY,   zz-locHZ+1)
                                vel_tmp(10) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+1)
                            end if
                                col_tmp(11) = label(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                                vel_tmp(11) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(12) = label(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                                vel_tmp(12) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(13) = label(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                                vel_tmp(13) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                            end if
                                col_tmp(14) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(14) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(15) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(15) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(16) = label(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                                vel_tmp(16) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                            end if
                                col_tmp(17) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(17) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(18) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(18) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(19) = label(xx-locHX,   yy-locHY,   zz-locHZ+2)
                                vel_tmp(19) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+2)
                            end if
                                col_tmp(20) = label(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                                vel_tmp(20) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(21) = label(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                                vel_tmp(21) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(22) = label(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                                vel_tmp(22) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                            end if
                                col_tmp(23) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(23) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(24) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(24) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(25) = label(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                                vel_tmp(25) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                            end if
                                col_tmp(26) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(26) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(27) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(27) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                            end if
                        end if
                    end if

                    ! rho_tmp
                    rho_tmp = ZERO
                    rho_tmp(1)  = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(2)  = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(3)  = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(4)  = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(5)  = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(6)  = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(7)  = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(8)  = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(9)  = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(10) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(11) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(12) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(13) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(14) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(15) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(16) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(17) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(18) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(19) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(20) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(21) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(22) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(23) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(24) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(25) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(26) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(27) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+3)
                    
                    ! value preprocessing
                    ee = epslon(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    dd =  delta(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    Cxx = -ONE - 2.0*ee
                    Cyy = -ONE - 2.0*ee
                    Czz = -ONE
                    Cxy = ZERO
                    Cyz = ZERO
                    Czx = ZERO

                    ! rhoX, rhoY and rhoZ preprocessing
                    rhoX = ZERO
                    rhoX(1) = (1.0/(rho_tmp(4) *Sx(xx-1)) + 1.0/(rho_tmp(5) *Sx(xx))) / 2.0
                    rhoX(2) = (1.0/(rho_tmp(6) *Sx(xx+1)) + 1.0/(rho_tmp(5) *Sx(xx))) / 2.0
                    rhoX(3) = (1.0/(rho_tmp(13)*Sx(xx-1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    rhoX(4) = (1.0/(rho_tmp(15)*Sx(xx+1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    rhoX(5) = (1.0/(rho_tmp(22)*Sx(xx-1)) + 1.0/(rho_tmp(23)*Sx(xx))) / 2.0
                    rhoX(6) = (1.0/(rho_tmp(24)*Sx(xx+1)) + 1.0/(rho_tmp(23)*Sx(xx))) / 2.0
                    rhoY = ZERO
                    rhoY(1) = (1.0/(rho_tmp(2) *Sy(yy-1)) + 1.0/(rho_tmp(5) *Sy(yy))) / 2.0
                    rhoY(2) = (1.0/(rho_tmp(8) *Sy(yy+1)) + 1.0/(rho_tmp(5) *Sy(yy))) / 2.0
                    rhoY(3) = (1.0/(rho_tmp(11)*Sy(yy-1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    rhoY(4) = (1.0/(rho_tmp(17)*Sy(yy+1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    rhoY(5) = (1.0/(rho_tmp(20)*Sy(yy-1)) + 1.0/(rho_tmp(23)*Sy(yy))) / 2.0
                    rhoY(6) = (1.0/(rho_tmp(26)*Sy(yy+1)) + 1.0/(rho_tmp(23)*Sy(yy))) / 2.0
                    rhoZ = ZERO
                    rhoZ(1) = (1.0/(rho_tmp(5) *Sz(zz-1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    rhoZ(2) = (1.0/(rho_tmp(23)*Sz(zz+1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0

                    ! val_tmp
                    val_tmp = ZERO

                    ! 1. d^2/dx^2
                    tmp  = Cxx * rx**2.0 * rho_tmp(14) / Sx(xx)
                    val_tmp(13) = val_tmp(13) + tmp * rhoX(3)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoX(3) + rhoX(4))
                    val_tmp(15) = val_tmp(15) + tmp * rhoX(4)
            
                    ! 2. d^2/dy^2
                    tmp  = Cyy * ry**2.0 * rho_tmp(14) / Sy(yy)
                    val_tmp(11) = val_tmp(11) + tmp * rhoY(3)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoY(3) + rhoY(4))
                    val_tmp(17) = val_tmp(17) + tmp * rhoY(4)
            
                    ! 3. d^2/dz^2
                    tmp  = Czz * rho_tmp(14) / Sz(zz)
                    val_tmp(5)  = val_tmp(5)  + tmp * rhoZ(1)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoZ(1) + rhoZ(2))
                    val_tmp(23) = val_tmp(23) + tmp * rhoZ(2)
            
                    ! 4. d^2/dxdy
                    tmp = Cxy * rx*ry/4.0 * rho_tmp(14) / (Sx(xx)*Sy(yy))
                    val_tmp(10) = val_tmp(10) + tmp * (rho_tmp(11)+rho_tmp(13))/(2.0*rho_tmp(11)*rho_tmp(13))
                    val_tmp(12) = val_tmp(12) - tmp * (rho_tmp(11)+rho_tmp(15))/(2.0*rho_tmp(11)*rho_tmp(15))
                    val_tmp(16) = val_tmp(16) - tmp * (rho_tmp(17)+rho_tmp(13))/(2.0*rho_tmp(17)*rho_tmp(13))
                    val_tmp(18) = val_tmp(18) + tmp * (rho_tmp(17)+rho_tmp(15))/(2.0*rho_tmp(17)*rho_tmp(15))
            
                    ! 5. d^2/dydz
                    tmp = Cyz * ry/4.0 * rho_tmp(14) / (Sy(yy)*Sz(zz))
                    val_tmp(2)  = val_tmp(2)  + tmp * (rho_tmp(5) +rho_tmp(11))/(2.0*rho_tmp(5) *rho_tmp(11))
                    val_tmp(8)  = val_tmp(8)  - tmp * (rho_tmp(5) +rho_tmp(17))/(2.0*rho_tmp(5) *rho_tmp(17))
                    val_tmp(20) = val_tmp(20) - tmp * (rho_tmp(23)+rho_tmp(11))/(2.0*rho_tmp(23)*rho_tmp(11))
                    val_tmp(26) = val_tmp(26) + tmp * (rho_tmp(23)+rho_tmp(17))/(2.0*rho_tmp(23)*rho_tmp(17))
            
                    ! 6. d^2/dzdx
                    tmp = Czx * rx/4.0 * rho_tmp(14) / (Sz(zz)*Sx(xx))
                    val_tmp(4)  = val_tmp(4)  + tmp * (rho_tmp(5) +rho_tmp(13))/(2.0*rho_tmp(5) *rho_tmp(13))
                    val_tmp(6)  = val_tmp(6)  - tmp * (rho_tmp(5) +rho_tmp(15))/(2.0*rho_tmp(5) *rho_tmp(15))
                    val_tmp(22) = val_tmp(22) - tmp * (rho_tmp(23)+rho_tmp(13))/(2.0*rho_tmp(23)*rho_tmp(13))
                    val_tmp(24) = val_tmp(24) + tmp * (rho_tmp(23)+rho_tmp(15))/(2.0*rho_tmp(23)*rho_tmp(15))
           
                    ! 7. d^4/dz^2dx^2
                    tmp  = -2.0*(ee-dd) * rx**2.0 * rho_tmp(14) / ( (hz*freq/vel_tmp(14))**2.0 * Sz(zz) * Sx(xx) )
                    tmp1 = tmp * rho_tmp(5)  * rhoZ(1)
                    tmp2 = tmp * rho_tmp(14) * (-rhoZ(1)-rhoZ(2))
                    tmp3 = tmp * rho_tmp(23) * rhoZ(2)
            
                    val_tmp(4)  = val_tmp(4)  + tmp1 * rhoX(1)
                    val_tmp(5)  = val_tmp(5)  - tmp1 *(rhoX(1)+rhoX(2))
                    val_tmp(6)  = val_tmp(6)  + tmp1 * rhoX(2)
                    val_tmp(13) = val_tmp(13) + tmp2 * rhoX(3)
                    val_tmp(14) = val_tmp(14) - tmp2 *(rhoX(3)+rhoX(4))
                    val_tmp(15) = val_tmp(15) + tmp2 * rhoX(4)
                    val_tmp(22) = val_tmp(22) + tmp3 * rhoX(5)
                    val_tmp(23) = val_tmp(23) - tmp3 *(rhoX(5)+rhoX(6))
                    val_tmp(24) = val_tmp(24) + tmp3 * rhoX(6)
            
                    ! 8. d^4/dz^2dy^2
                    tmp  = -2.0*(ee-dd) * ry**2.0 * rho_tmp(14) / ( (hz*freq/vel_tmp(14))**2.0 * Sz(zz) * Sy(yy) )
                    tmp1 = tmp * rho_tmp(5)  * rhoZ(1)
                    tmp2 = tmp * rho_tmp(14) * (-rhoZ(1)-rhoZ(2))
                    tmp3 = tmp * rho_tmp(23) * rhoZ(2)
            
                    val_tmp(2)  = val_tmp(2)  + tmp1 * rhoY(1)
                    val_tmp(5)  = val_tmp(5)  - tmp1 *(rhoY(1)+rhoY(2))
                    val_tmp(8)  = val_tmp(8)  + tmp1 * rhoY(2)
                    val_tmp(11) = val_tmp(11) + tmp2 * rhoY(3)
                    val_tmp(14) = val_tmp(14) - tmp2 *(rhoY(3)+rhoY(4))
                    val_tmp(17) = val_tmp(17) + tmp2 * rhoY(4)
                    val_tmp(20) = val_tmp(20) + tmp3 * rhoY(5)
                    val_tmp(23) = val_tmp(23) - tmp3 *(rhoY(5)+rhoY(6))
                    val_tmp(26) = val_tmp(26) + tmp3 * rhoY(6)
 
                    ! 7. mass term
                    val_tmp(14) = val_tmp(14) - (hz*freq/vel_tmp(14))**2.0


                    ! 8. KEY step: modification for the sake of overlapping interface
                    if (locHX > 2 .and. xx == locHX) then
                        val_tmp(2:26:3) = ZERO
                    end if
                    if (locHY > 2 .and. yy == locHY) then
                        val_tmp(4:6)   = ZERO
                        val_tmp(13:15) = ZERO
                        val_tmp(22:24) = ZERO                       
                    end if
                    if (locHZ > 2 .and. zz == locHZ) then
                        val_tmp(10:18) = ZERO                        
                    end if
                   
                    ! copy val and col
                    counter = 0
                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    do kk = 1,27
                        if (col_tmp(kk) /= 0) then
                            A%col( A%ind(lvl)+counter ) = col_tmp(kk)
                            A%val( A%ind(lvl)+counter ) = val_tmp(kk)
                            counter = counter + 1
                        end if
                    end do

                end do
            end do
        end do
        deallocate(Sx,Sy,Sz)

        return
    end subroutine A3D_VTI


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  HTI  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    subroutine A3D_HTI( freq, Nx, Ny, Nz, N_delta, sigma0, hx, hy, hz, locHX, locTX, &
         locHY, locTY, locHZ, locTZ, label, vel, rho, epslon, delta,  a_ind, a_col, a_val )

        complex(kind=4) :: freq, sigma0, hx, hy, hz
        integer         :: Nx, Ny, Nz, N_delta, locHX, locTX, locHY, locTY, locHZ, locTZ
        integer         ::  label(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    vel(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::    rho(locTX-locHX+3,locTY-locHY+3,locTZ-locHZ+3)
        real(kind=4)    :: epslon(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        real(kind=4)    ::  delta(locTX-locHX+1,locTY-locHY+1,locTZ-locHZ+1)
        integer, allocatable :: a_ind(:), a_col(:)
        complex(kind=4), allocatable :: a_val(:)
!        type(CSR)       :: A

        integer                     :: xx, yy, zz, kk, lvl, length, counter, col_tmp(27)
        complex(kind=4)             :: rx, ry, Cxx, Cyy, Czz, Cxy, Cyz, Czx, ee, dd, tmp
        complex(kind=4)             :: tmp1, tmp2, tmp3
        complex(kind=4)             :: rhoX(2), rhoY(2), rhoZX(6), rhoZY(6)
        complex(kind=4)             :: vel_tmp(27), rho_tmp(27), val_tmp(27)
        complex(kind=4),allocatable :: Sx(:), Sy(:), Sz(:)

        ! the parameters
        rx  = hz / hx
        ry  = hz / hy

        ! generating Sx, Sy and Sz
        allocate(Sx(Nx),Sy(Ny),Sz(Nz))
        Sx = ONE
        Sy = ONE
        Sz = ONE
        do counter = 1,N_delta
            tmp = RT * (sigma0/freq) * ( 1.0 + cos(real(pi)*real(counter)/real(N_delta)) ) / 2.0    ! here we are using a cos^2 taper function
            Sx(counter)      = Sx(counter) - tmp
            Sx(Nx-counter+1) = Sx(counter)
            Sy(counter)      = Sy(counter) - tmp
            Sy(Ny-counter+1) = Sy(counter)
            Sz(counter)      = Sz(counter) - tmp
            Sz(Nz-counter+1) = Sz(counter)      
        end do

        ! allocating A%ind
        length = (locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1)    ! the order of A
        allocate(a_ind(length+1))
        a_ind = 0

        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1) + 1
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                                a_ind(lvl) = a_ind(lvl) + 1
                            if (xx < locTX) then
                                a_ind(lvl) = a_ind(lvl) + 1
                            end if
                        end if
                    end if

                end do
            end do
        end do

        a_ind(1) = 1
        do kk = 1,length
            a_ind(kk+1) = a_ind(kk+1) + a_ind(kk)
        end do

        ! allocating A%col and A%val
        length = 27*(locTX-locHX+1)*(locTY-locHY+1)*(locTZ-locHZ+1) &
             - 18*((locTX-locHX-1)*(locTY-locHY-1)+(locTY-locHY-1)*(locTZ-locHZ-1)&
             +(locTZ-locHZ-1)*(locTX-locHX-1)) - 60*((locTX-locHX-1)+(locTY-locHY-1)&
             +(locTZ-locHZ-1)) - 19*8
        allocate(A%col(length),A%val(length))
        a_col = 0
        a_val = ZERO

        ! generating A
        do zz = locHZ,locTZ
            do yy = locHY,locTY
                do xx = locHX,locTX

                    ! col_tmp and vel_tmp
                    col_tmp = 0
                    vel_tmp = ZERO
                    if (zz > locHZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(1) = label(xx-locHX,   yy-locHY,   zz-locHZ)
                                vel_tmp(1) =   vel(xx-locHX,   yy-locHY,   zz-locHZ)
                            end if
                                col_tmp(2) = label(xx-locHX+1, yy-locHY,   zz-locHZ)
                                vel_tmp(2) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(3) = label(xx-locHX+2, yy-locHY,   zz-locHZ)
                                vel_tmp(3) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(4) = label(xx-locHX,   yy-locHY+1, zz-locHZ)
                                vel_tmp(4) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ)
                            end if
                                col_tmp(5) = label(xx-locHX+1, yy-locHY+1, zz-locHZ)
                                vel_tmp(5) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(6) = label(xx-locHX+2, yy-locHY+1, zz-locHZ)
                                vel_tmp(6) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(7) = label(xx-locHX,   yy-locHY+2, zz-locHZ)
                                vel_tmp(7) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ)
                            end if
                                col_tmp(8) = label(xx-locHX+1, yy-locHY+2, zz-locHZ)
                                vel_tmp(8) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ)
                            if (xx < locTX) then
                                col_tmp(9) = label(xx-locHX+2, yy-locHY+2, zz-locHZ)
                                vel_tmp(9) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ)
                            end if
                        end if                      
                    end if
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(10) = label(xx-locHX,   yy-locHY,   zz-locHZ+1)
                                vel_tmp(10) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+1)
                            end if
                                col_tmp(11) = label(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                                vel_tmp(11) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(12) = label(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                                vel_tmp(12) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+1)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(13) = label(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                                vel_tmp(13) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+1)
                            end if
                                col_tmp(14) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(14) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(15) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                                vel_tmp(15) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+1)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(16) = label(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                                vel_tmp(16) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+1)
                            end if
                                col_tmp(17) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(17) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+1)
                            if (xx < locTX) then
                                col_tmp(18) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                                vel_tmp(18) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+1)
                            end if
                        end if
                    if (zz < locTZ) then
                        if (yy > locHY) then
                            if (xx > locHX) then
                                col_tmp(19) = label(xx-locHX,   yy-locHY,   zz-locHZ+2)
                                vel_tmp(19) =   vel(xx-locHX,   yy-locHY,   zz-locHZ+2)
                            end if
                                col_tmp(20) = label(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                                vel_tmp(20) =   vel(xx-locHX+1, yy-locHY,   zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(21) = label(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                                vel_tmp(21) =   vel(xx-locHX+2, yy-locHY,   zz-locHZ+2)
                            end if
                        end if
                            if (xx > locHX) then
                                col_tmp(22) = label(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                                vel_tmp(22) =   vel(xx-locHX,   yy-locHY+1, zz-locHZ+2)
                            end if
                                col_tmp(23) = label(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(23) =   vel(xx-locHX+1, yy-locHY+1, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(24) = label(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                                vel_tmp(24) =   vel(xx-locHX+2, yy-locHY+1, zz-locHZ+2)
                            end if    
                        if (yy < locTY) then
                            if (xx > locHX) then
                                col_tmp(25) = label(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                                vel_tmp(25) =   vel(xx-locHX,   yy-locHY+2, zz-locHZ+2)
                            end if
                                col_tmp(26) = label(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(26) =   vel(xx-locHX+1, yy-locHY+2, zz-locHZ+2)
                            if (xx < locTX) then
                                col_tmp(27) = label(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                                vel_tmp(27) =   vel(xx-locHX+2, yy-locHY+2, zz-locHZ+2)
                            end if
                        end if
                    end if

                    ! rho_tmp
                    rho_tmp = ZERO
                    rho_tmp(1)  = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(2)  = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(3)  = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+1)
                    rho_tmp(4)  = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(5)  = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(6)  = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+1)
                    rho_tmp(7)  = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(8)  = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(9)  = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+1)
                    rho_tmp(10) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(11) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(12) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+2)
                    rho_tmp(13) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(14) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(15) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+2)
                    rho_tmp(16) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(17) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(18) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+2)
                    rho_tmp(19) = rho(xx-locHX+1,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(20) = rho(xx-locHX+2,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(21) = rho(xx-locHX+3,yy-locHY+1,zz-locHZ+3)
                    rho_tmp(22) = rho(xx-locHX+1,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(23) = rho(xx-locHX+2,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(24) = rho(xx-locHX+3,yy-locHY+2,zz-locHZ+3)
                    rho_tmp(25) = rho(xx-locHX+1,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(26) = rho(xx-locHX+2,yy-locHY+3,zz-locHZ+3)
                    rho_tmp(27) = rho(xx-locHX+3,yy-locHY+3,zz-locHZ+3)
                    
                    ! value preprocessing
                    ee = epslon(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    dd =  delta(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    Cxx = -ONE
                    Cyy = -ONE
                    Czz = -ONE - 2.0*ee
                    Cxy = ZERO
                    Cyz = ZERO
                    Czx = ZERO

                    ! rhoX, rhoY and rhoZ preprocessing
                    rhoX = ZERO
                    rhoX(1) = (1.0/(rho_tmp(13)*Sx(xx-1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    rhoX(2) = (1.0/(rho_tmp(15)*Sx(xx+1)) + 1.0/(rho_tmp(14)*Sx(xx))) / 2.0
                    rhoY = ZERO
                    rhoY(1) = (1.0/(rho_tmp(11)*Sy(yy-1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    rhoY(2) = (1.0/(rho_tmp(17)*Sy(yy+1)) + 1.0/(rho_tmp(14)*Sy(yy))) / 2.0
                    rhoZX = ZERO
                    rhoZX(1) = (1.0/(rho_tmp(4) *Sz(zz-1)) + 1.0/(rho_tmp(13)*Sz(zz))) / 2.0
                    rhoZX(2) = (1.0/(rho_tmp(22)*Sz(zz+1)) + 1.0/(rho_tmp(13)*Sz(zz))) / 2.0
                    rhoZX(3) = (1.0/(rho_tmp(5) *Sz(zz-1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    rhoZX(4) = (1.0/(rho_tmp(23)*Sz(zz+1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    rhoZX(5) = (1.0/(rho_tmp(6) *Sz(zz-1)) + 1.0/(rho_tmp(15)*Sz(zz))) / 2.0
                    rhoZX(6) = (1.0/(rho_tmp(24)*Sz(zz+1)) + 1.0/(rho_tmp(15)*Sz(zz))) / 2.0
                    rhoZY = ZERO
                    rhoZY(1) = (1.0/(rho_tmp(2) *Sz(zz-1)) + 1.0/(rho_tmp(11)*Sz(zz))) / 2.0
                    rhoZY(2) = (1.0/(rho_tmp(20)*Sz(zz+1)) + 1.0/(rho_tmp(11)*Sz(zz))) / 2.0
                    rhoZY(3) = (1.0/(rho_tmp(5) *Sz(zz-1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    rhoZY(4) = (1.0/(rho_tmp(23)*Sz(zz+1)) + 1.0/(rho_tmp(14)*Sz(zz))) / 2.0
                    rhoZY(5) = (1.0/(rho_tmp(8) *Sz(zz-1)) + 1.0/(rho_tmp(17)*Sz(zz))) / 2.0
                    rhoZY(6) = (1.0/(rho_tmp(26)*Sz(zz+1)) + 1.0/(rho_tmp(17)*Sz(zz))) / 2.0

                    ! val_tmp
                    val_tmp = ZERO

                    ! 1. d^2/dx^2
                    tmp  = Cxx * rx**2.0 * rho_tmp(14) / Sx(xx)
                    val_tmp(13) = val_tmp(13) + tmp * rhoX(1)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoX(1) + rhoX(2))
                    val_tmp(15) = val_tmp(15) + tmp * rhoX(2)
            
                    ! 2. d^2/dy^2
                    tmp  = Cyy * ry**2.0 * rho_tmp(14) / Sy(yy)
                    val_tmp(11) = val_tmp(11) + tmp * rhoY(1)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoY(1) + rhoY(2))
                    val_tmp(17) = val_tmp(17) + tmp * rhoY(2)
            
                    ! 3. d^2/dz^2
                    tmp  = Czz * rho_tmp(14) / Sz(zz)
                    val_tmp(5)  = val_tmp(5)  + tmp * rhoZX(3)
                    val_tmp(14) = val_tmp(14) - tmp *(rhoZX(3) + rhoZX(4))
                    val_tmp(23) = val_tmp(23) + tmp * rhoZX(4)
            
                    ! 4. d^2/dxdy
                    tmp = Cxy * rx*ry/4.0 * rho_tmp(14) / (Sx(xx)*Sy(yy))
                    val_tmp(10) = val_tmp(10) + tmp * (rho_tmp(11)+rho_tmp(13))/(2.0*rho_tmp(11)*rho_tmp(13))
                    val_tmp(12) = val_tmp(12) - tmp * (rho_tmp(11)+rho_tmp(15))/(2.0*rho_tmp(11)*rho_tmp(15))
                    val_tmp(16) = val_tmp(16) - tmp * (rho_tmp(17)+rho_tmp(13))/(2.0*rho_tmp(17)*rho_tmp(13))
                    val_tmp(18) = val_tmp(18) + tmp * (rho_tmp(17)+rho_tmp(15))/(2.0*rho_tmp(17)*rho_tmp(15))
            
                    ! 5. d^2/dydz
                    tmp = Cyz * ry/4.0 * rho_tmp(14) / (Sy(yy)*Sz(zz))
                    val_tmp(2)  = val_tmp(2)  + tmp * (rho_tmp(5) +rho_tmp(11))/(2.0*rho_tmp(5) *rho_tmp(11))
                    val_tmp(8)  = val_tmp(8)  - tmp * (rho_tmp(5) +rho_tmp(17))/(2.0*rho_tmp(5) *rho_tmp(17))
                    val_tmp(20) = val_tmp(20) - tmp * (rho_tmp(23)+rho_tmp(11))/(2.0*rho_tmp(23)*rho_tmp(11))
                    val_tmp(26) = val_tmp(26) + tmp * (rho_tmp(23)+rho_tmp(17))/(2.0*rho_tmp(23)*rho_tmp(17))
            
                    ! 6. d^2/dzdx
                    tmp = Czx * rx/4.0 * rho_tmp(14) / (Sz(zz)*Sx(xx))
                    val_tmp(4)  = val_tmp(4)  + tmp * (rho_tmp(5) +rho_tmp(13))/(2.0*rho_tmp(5) *rho_tmp(13))
                    val_tmp(6)  = val_tmp(6)  - tmp * (rho_tmp(5) +rho_tmp(15))/(2.0*rho_tmp(5) *rho_tmp(15))
                    val_tmp(22) = val_tmp(22) - tmp * (rho_tmp(23)+rho_tmp(13))/(2.0*rho_tmp(23)*rho_tmp(13))
                    val_tmp(24) = val_tmp(24) + tmp * (rho_tmp(23)+rho_tmp(15))/(2.0*rho_tmp(23)*rho_tmp(15))
           
                    ! 7. d^4/dx^2dz^2
                    tmp  = -2.0*(ee-dd) * rx**2.0 * rho_tmp(14) / ( (hz*freq/vel_tmp(14))**2.0 * Sz(zz) * Sx(xx) )
                    tmp1 = tmp * rho_tmp(13) * rhoX(1)
                    tmp2 = tmp * rho_tmp(14) * (-rhoX(1)-rhoX(2))
                    tmp3 = tmp * rho_tmp(15) * rhoX(2)
            
                    val_tmp(4)  = val_tmp(4)  + tmp1 * rhoZX(1)
                    val_tmp(13) = val_tmp(13) - tmp1 *(rhoZX(1)+rhoZX(2))
                    val_tmp(22) = val_tmp(22) + tmp1 * rhoZX(2)
                    val_tmp(5)  = val_tmp(5)  + tmp2 * rhoZX(3)
                    val_tmp(14) = val_tmp(14) - tmp2 *(rhoZX(3)+rhoZX(4))
                    val_tmp(23) = val_tmp(23) + tmp2 * rhoZX(4)
                    val_tmp(6)  = val_tmp(6)  + tmp3 * rhoZX(5)
                    val_tmp(15) = val_tmp(15) - tmp3 *(rhoZX(5)+rhoZX(6))
                    val_tmp(24) = val_tmp(24) + tmp3 * rhoZX(6)
            
                    ! 8. d^4/dy^2dz^2
                    tmp  = -2.0*(ee-dd) * ry**2.0 * rho_tmp(14) / ( (hz*freq/vel_tmp(14))**2.0 * Sz(zz) * Sy(yy) )
                    tmp1 = tmp * rho_tmp(11) * rhoY(1)
                    tmp2 = tmp * rho_tmp(14) * (-rhoY(1)-rhoY(2))
                    tmp3 = tmp * rho_tmp(17) * rhoY(2)
            
                    val_tmp(2)  = val_tmp(2)  + tmp1 * rhoZY(1)
                    val_tmp(11) = val_tmp(11) - tmp1 *(rhoZY(1)+rhoZY(2))
                    val_tmp(20) = val_tmp(20) + tmp1 * rhoZY(2)
                    val_tmp(5)  = val_tmp(5)  + tmp2 * rhoZY(3)
                    val_tmp(14) = val_tmp(14) - tmp2 *(rhoZY(3)+rhoZY(4))
                    val_tmp(23) = val_tmp(23) + tmp2 * rhoZY(4)
                    val_tmp(8)  = val_tmp(8)  + tmp3 * rhoZY(5)
                    val_tmp(17) = val_tmp(17) - tmp3 *(rhoZY(5)+rhoZY(6))
                    val_tmp(26) = val_tmp(26) + tmp3 * rhoZY(6)
 
                    ! 7. mass term
                    val_tmp(14) = val_tmp(14) - (hz*freq/vel_tmp(14))**2.0


                    ! 8. KEY step: modification for the sake of overlapping interface
                    if (locHX > 2 .and. xx == locHX) then
                        val_tmp(2:26:3) = ZERO
                    end if
                    if (locHY > 2 .and. yy == locHY) then
                        val_tmp(4:6)   = ZERO
                        val_tmp(13:15) = ZERO
                        val_tmp(22:24) = ZERO                       
                    end if
                    if (locHZ > 2 .and. zz == locHZ) then
                        val_tmp(10:18) = ZERO                        
                    end if
                   
                    ! copy val and col
                    counter = 0
                    lvl = label(xx-locHX+1,yy-locHY+1,zz-locHZ+1)
                    do kk = 1,27
                        if (col_tmp(kk) /= 0) then
                            a_col( a_ind(lvl)+counter ) = col_tmp(kk)
                            a_val( a_ind(lvl)+counter ) = val_tmp(kk)
                            counter = counter + 1
                        end if
                    end do

                end do
            end do
        end do
        deallocate(Sx,Sy,Sz)

        return
    end subroutine A3D_HTI


end module A3D
