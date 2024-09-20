#### script to download planet images in command line 




from planetscopeAPI import PlanetScopeOrdersAPI
API = PlanetScopeOrdersAPI.PlanetScopeAPIOrder(selectSites=False, threading = False, printPolling=True) # initalizing the class variable
API.get_all_data()