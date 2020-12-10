# itd_insert

insert ITD from another model or source into GEOS5 sea ice restart

only works with GEOS-5 style sea ice (thermo) restart files 



*  `itd_insert_fast.py`: not optimized yet, runs a bit slow

## Usage

    `itd_insert_fast.py` expects 4 command line arguments
       
      * `tilefile`, GEOS coupled model provides
      * `institute/source name`, only 'gfdl' and 'mit' currently supported
      * `seaice restart filename from above institute`
      * `seaicethermo_internal_rst from GEOS`, optional
       
