# itd_insert

insert ITD from another model or source into GEOS5 sea ice restart

only works with GEOS-5 style sea ice (thermo) restart files 



*  `itd_insert_fast.py`: not optimized yet, runs a bit slow

## Usage

`itd_insert_fast.py` expects 4 command line arguments
       
-  `tilefile`, GEOS coupled model provides
-  `institute/source name`, only `gfdl` and `mit` currently supported
-  `seaice restart filename` from above institute
-  `seaicethermo_internal_rst` or `saltwater_internal_rst`, this is the 
    template restart file from GEOS coupled model; it doesn't have to match 
    the size of the target tile file; 2 versions are provided (not
    in the repo)
       
