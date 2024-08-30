import csv
import numpy as np
import pandas as ps
from pyomo.environ import *
import math

# Multi-objective variables
weight = 1 # weighting for cost objective (0 to 1)
tax = 31.84 # carbon tax in the uk £/tonne (affects weighting of emissions)


# Input variables
currentBoiler = 2 # current type of boiler in place (0 for gas, 1 for electric, 2 for oil)
BrewDaysSpring = 56 # total number of brewing days in spring
BrewDaysSummer = 56 # et cetera
BrewDaysAutumn = 48
BrewDaysWinter = 48
Tmash = 66 # desired temperature for mashing
Tlaut = 80 # desired temperature for lautering
Tboil = 100 # desired temperature for boiling
To_spring = 10 # Average ambient temperature in spring
To_summer = 16 # Average ambient temperature in summer
To_autumn = 10 # Average ambient temperature in autumn
To_winter = 5 # Average ambient temperature in winter
SunlightArea = 100 # surface area with sunlight exposure (e.g. on rooftops) in m^2
BSGproduction = 1500 # mass of BSG available for digestion in kg (produced per batch or stored)
non_brew = 4.4 # average hourly electricity usage on non-brewing days (kW)
ADspace = 10 # space available at the brewery for an anaerobic digester (m^3)

To = [To_spring, To_summer, To_autumn, To_winter]

# Importing data
spring = ps.DataFrame(data = csv.reader(open("Spring1.csv")))
summer = ps.DataFrame(data = csv.reader(open("Summer1.csv")))
autumn = ps.DataFrame(data = csv.reader(open("Autumn1.csv")))
winter = ps.DataFrame(data = csv.reader(open("Winter1.csv")))
wind = ps.DataFrame(data = csv.reader(open("Windspeeddata.csv")))

# PV Panel information
P_panel = 380/1000 # kiloWatts
panel_length = 1.765 # metres
panel_width = 1.048 # metres
a = 0.0045
NOCT = 45
max_number = SunlightArea/(panel_length*panel_width) # max number of PV panels

# Wind turbine information
radius = 4.5 # radius of each blade
cp = 0.45 # coefficient of performance

iindex = 5
# Calculating max power output of PV per panel based on local irradiance
irradiance = ps.DataFrame(data = (spring[iindex], summer[iindex], autumn[iindex], winter[iindex]), dtype = float)
temperature = ps.DataFrame(data = (spring[7], summer[7], autumn[7], winter[7]), dtype = float)
windspeed = ps.DataFrame(data = (wind[0], wind[1], wind[2], wind[3]), dtype = float)


# Converting DataFrames to numpy arrays
irradiance = np.transpose(irradiance.to_numpy())
temperature = np.transpose(temperature.to_numpy())
windspeed = windspeed.to_numpy()

# Calculating T_c based on equation 2 (see Ruosi's documentation)
T_c = irradiance/800*(NOCT-20) + temperature
P_PV = P_panel * irradiance / 1000*(1-a*(T_c -25))

windSpeed = windspeed/2.237 # wind speed in m/s rather than mph
windPower = cp*radius**2*windSpeed**3*0.5*1.225/1000*math.pi # wind turbine power output in kW
turbineMax = np.max(windPower)

investment_panel = 0.3132*1000*P_panel*4
investment_battery = 900
investment_boiler_gas = 28964+1625+535+886+65
investment_boiler_ele = 24144+1625+535+886+65
investment_boiler_oil = 28964+1625+535+886+65
investment_turbine = 5675*0.81


# Opening load demand data
load_demand_ele = ps.read_excel("load_demand.xlsx", "Sheet3")
load_demand_ele = load_demand_ele.to_numpy()
r = np.r_[0:(len(load_demand_ele)-4):4]
load_ele = load_demand_ele[r,3]
load_demand_gas = ps.read_excel("load_demand.xlsx", "Sheet2")
load_demand_gas = load_demand_gas.to_numpy()
r = np.r_[0:(len(load_demand_gas)-4):4]
load_gas = load_demand_gas[r,1]
load = ps.DataFrame.transpose(ps.DataFrame([load_gas, load_ele]))
load = load.to_numpy()

# Basic parameters
T = 24
day = [90, 92, 92, 91] # number of days in four seasons
price_ele = 2105.82/11262 # price of electricity
sell_ele = 5/100 # price of feed-in tariff

# Battery parameters
nominal_cap = 3552/1000
charge_eff = 0.90
discharge_eff = 0.95

# Total thermal demand based on Ruosi's calculation per batch
total_heat = 2774
MJ_kWh = 0.277778
gas_price = 7.4/100
oil_price = 10.4/100

# Data for CCS with DFM
CO2perkWh = 0.185 # carbon dioxide produced in kg per kWh of natural gas burnt
CO2oil = 0.245 # carbon dioxide produced in kg per kWh of oil burnt
price_H2 = 5.62 # cost of green hydrogen per kg
CH4enthalpy = 890.1 # enthalpy of combustion of methane in kJ/mol
WHSV = 24 # weight hourly space velocity tested for NiRu catalyst in L/g/h
catalystPrice = 3.81 # estimated cost of catalyst in £/g based on my calculations
BrewDaysSum = BrewDaysSpring + BrewDaysSummer + BrewDaysWinter + BrewDaysAutumn

# Heat loss estimations
AverageTime = sum(day)/BrewDaysSum*24*3600
Tfinal = np.r_[0:len(To)]
for i in np.r_[0:len(To)]:
    Tfinal[i] = To[i]+(90-To[i])*math.exp(-0.024*(10**(-3))/(4.2*50)*AverageTime)
    Tfinalavg = np.average(Tfinal)

therm = abs(load[:,0] - load[:,1]) # thermal demand

U_DFM = 2.37*10**(-5) # overall heat transfer coefficient using rockwool for DFM in kW m^-2 K^-1
UAD = 2.37*10**(-5) # overall heat transfer coeffcient using rockwool for AD in kW m^-2 K^-1

# thermal load timeframe calculations
non_zero = np.count_nonzero(load[:,1] - load[:,0])
load_profile = np.r_[0:24]
non_zeroes = np.nonzero(load[:,1] - load[:,0])
first = non_zeroes[0]
first_index = first[0]

# Variables for anaerobic digestion
Tmeso = 35 # Optimal temperature for mesophilic bacteria
Tthermo = 55 # Optimal temperature for thermophilic bacteria
MesoCH4 = 163/1500*0.55 # Mesophilic methane production l/d
ThermoCH4 = 163/1500*0.65 # Thermophilic methane production m^3/kg/d
MesoBiogas = 2.697 # Mesophilic total biogas production l/d
ThermoBiogas = 184/1500 # Thermophilic total biogas production m^3/kg/d
DaysUntilReady = 25 # the reference I found let digestion occur over 25 days
BSGdensity = 432 # density of brewer's spent grains in kg/m^3

# AD calculations
ADDays = math.floor((365.25)/DaysUntilReady) # number of days on which biogas can be used
MethaneCombustionEnthalpy = 890.63 # Enthalpy of (complete) combustion of methane in kJ/mol
MesoMols = 8.3143*Tmeso/101325/(MesoCH4/1000) # Mols of methane from mesophilic digestion
ThermoMols = 8.3143*Tthermo/101325/(ThermoCH4/1000) # Mols of methane from thermophilic digestion
MesoMJ = MethaneCombustionEnthalpy*MesoMols/1000 # Total heat from burning mesophilic biogas
ThermoMJ = MethaneCombustionEnthalpy*ThermoMols/1000 # Total heat from burning thermophilic biogas

# Constants and data for heat storage
H = 2300 # Specific heat of evaporation/condensation [kJ/kg]
Cp_wort = 4 # Average specific heat of wort [kJ/kg/K]
Cp_water = 4.2 # Average specific heat of water [kJ/kg/K]
T_store = 90 # desired temperature for heat storage (Celsius)

V_wort = 10**3 # Amount of wort from sparging unit to copper (L or kg)
Coef_steam = 1 # Average ratio of steam from copper
V_steam = 10000

price_insul = 20  # Approximate maximum average cost of insulation material [pounds/m2]
U_hex = 5.916 # Heat transfer coefficient for heat storage system (kW m^-2 K^-1)

CEPCI_2010 = 532.9
CEPCI_2023 = 800.8


therm = abs(load[:,0] - load[:,1])
maxCap = 72.61 # maximum capacity of the DFM reactor in m^3

lifetime = [7]


for f in np.r_[0:len(lifetime)]:
    b = ConcreteModel()
    b.i = Set(initialize = np.r_[0:8]) # indices for days (0-3 are brewing days and 4-7 are non-brewing)
    b.j = Set(initialize = np.r_[0:T]) # indices for hours in the day
    b.m = Set(initialize = np.r_[0:max_number]) # indices for PV system
    b.k = Set(initialize = np.r_[0:4]) # indices for brewing days only
    b.l = Set(initialize = np.r_[4:8]) # indices for non-brewing days only
    
    # Variables for boiler selection, PV cells, wind and battery storage
    b.p_solar = Var(b.i, b.j, domain = NonNegativeReals) # PV Power
    b.num_solar = Var(b.m, domain = Binary) # Number of PV panels
    b.p_buy = Var(b.i, b.j, domain = NonNegativeReals) # Electricity bought from grid
    b.p_sell = Var(b.i,b.j, domain = NonNegativeReals) # Electricity sold to grid
    b.p_main = Var(b.i, b.j, domain = Reals) # Power from grid
    b.K_main = Var(b.i,b.j,domain = Binary) # Binary variable defining when electricity is bought or sold
    b.investment = Var(domain = NonNegativeReals) # Total capital costs
    b.K_boiler = Var(Set(initialize = np.r_[0:3]), domain = Binary) # Binary variables defining boiler type
    b.SOC = Var(b.i, b.j, domain = Reals, bounds = (5, 280)) # State of charge
    b.p_dis = Var(b.i, b.j, domain = NonNegativeReals) # Discharge power of storage
    b.p_cha = Var(b.i, b.j, domain = NonNegativeReals) # Charge power of storage
    b.p_ess = Var(b.i, b.j, domain = Reals) # Power of storage
    b.K_ess = Var(b.i, b.j, domain = Binary) # Binary variable showing if energy is being charged or discharged from battery
    b.p_wind = Var(b.i, b.j, domain = NonNegativeReals, bounds = (0, turbineMax)) # Wind power
    b.turbine_capacity = Var(domain = NonNegativeReals, bounds = (0, turbineMax))
    b.K_wind = Var(domain = Binary)
    b.battery_capacity = Var(domain = NonNegativeReals, bounds = (0,10))
    
    # Variables for heat storage
    b.K_heat = Var(domain = Binary) # binary variable for implementation of the entire heat storage system
    b.K_HEX = Var(Set(initialize = np.r_[0:2]), domain = Binary) # binary variables for selecting the optimal heat exchangers
    b.V_water_mash = Var(domain = NonNegativeReals) # volume of water stored for heat exchange after mashing
    b.V_water_laut = Var(domain = NonNegativeReals) # volume of water stored for heat exchanger after lautering
    b.A_boil = Var(domain = NonNegativeReals, bounds = (0,20)) # area of heat exchanger after boiling (required for heat storage)
    b.A_mash = Var(domain = NonNegativeReals, bounds = (0,20)) # area of heat exchanger after mashing (optional)
    b.A_laut = Var(domain = NonNegativeReals, bounds = (0,20)) # area of heat exchanger after lautering (optional)
    b.HexCapex = Var(domain = NonNegativeReals) # total capital costs of heat storage
    b.T_post_mash = Var(domain = NonNegativeReals, bounds = (Tmash, T_store - 5)) # temperature after heating after mash tun
    b.T_post_laut = Var(domain = NonNegativeReals, bounds = (Tlaut, T_store - 5)) # temperature after heating after lauter tun
    b.T_post_mash_avg = Var(domain = NonNegativeReals, bounds = (Tmash, T_store - 5))
    b.T_post_laut_avg = Var(domain = NonNegativeReals, bounds = (Tmash, T_store - 5))
    b.Q_savings_1 = Var(domain = NonNegativeReals) # savings from mashing heat exchanger
    b.Q_savings_2 = Var(domain = NonNegativeReals) # savings from lautering heat exchanger
    b.Q_savings_1_gas = Var(domain = NonNegativeReals)
    b.Q_savings_2_gas = Var(domain = NonNegativeReals)
    b.Q_savings_1_ele = Var(domain = NonNegativeReals) # same savings but implemented into thermal load
    b.Q_savings_2_ele = Var(domain = NonNegativeReals)
    b.T_store_new = Var(domain = NonNegativeReals, bounds = (60, T_store)) # storage temperature after heating post-mashing and/or post-lautering
    

    DeltaTboil = Tboil - Tmash
    
    DeltaTmash = np.r_[0:4]
    DeltaTlaut = np.r_[0:4]
    
    DeltaTmash = np.average(Tfinal) - Tmash
    DeltaTlaut = np.average(Tfinal) - Tlaut
    DeltaT2 = 5
    
    t_boil = 60*15 # time needed to contact on boiling heat exchanger (seconds)
    t_mash = 60*15 # time needed to contact on mashing heat exchanger (seconds)
    t_laut = 60*15 # time needed to contact on lautering heat exchanger (seconds)
    
    # log mean temperatures
    alphaBoil = DeltaTboil/DeltaT2
    alphaMash = DeltaTmash/DeltaT2
    alphaLaut = DeltaTlaut/DeltaT2
    epsilon = 10**-5
    LogMeanBoil = DeltaT2*((alphaBoil-1)**2+epsilon)**0.5/(log(alphaBoil)**2+epsilon)**0.5
    LogMeanMash = DeltaT2*((alphaMash-1)**2+epsilon)**0.5/(log(alphaMash)**2+epsilon)**0.5
    LogMeanLaut = DeltaT2*((alphaLaut-1)**2+epsilon)**0.5/(log(alphaLaut)**2+epsilon)**0.5
    
    b.HexHeat1 = Constraint(expr = (b.V_water_mash + b.V_water_laut)*Cp_water*(T_store - Tmash) <= V_steam*H)
    b.HexHeat2 = Constraint(expr = b.V_water_mash*Cp_water*(T_store - b.T_post_mash+5) >= V_wort*Cp_wort*(b.T_post_mash-Tmash))
    b.HexHeat3 = Constraint(expr = b.V_water_laut*Cp_water*(T_store - b.T_post_laut+5) >= V_wort*Cp_wort*(b.T_post_laut-Tlaut))
    
    DeltaTboil = Tboil - Tmash
    DeltaTmash = T_store - Tmash
    DeltaTlaut = T_store - Tlaut
    DeltaT2 = 5
    
    LogMeanBoil = (DeltaTboil - DeltaT2)/log(DeltaTboil/DeltaT2)
    LogMeanMash = (DeltaTmash - DeltaT2)/log(DeltaTmash/DeltaT2)
    LogMeanLaut = (DeltaTlaut - DeltaT2)/log(DeltaTlaut/DeltaT2)
    
    b.Q1 = Constraint(expr = b.Q_savings_1 == V_wort*Cp_wort*(b.T_post_mash-Tmash)/100) # heat savings in MJ
    b.Q2 = Constraint(expr = b.Q_savings_2 == V_wort*Cp_wort*(b.T_post_laut-Tlaut)/100)
    b.Q1again = Constraint(expr = b.Q_savings_1_gas <= b.Q_savings_1)
    b.Q2again = Constraint(expr = b.Q_savings_1_gas <= b.Q_savings_1)
    b.Q1ele = Constraint(expr = b.Q_savings_1_ele <= b.Q_savings_1*MJ_kWh)
    b.Q2ele = Constraint(expr = b.Q_savings_2_ele <= b.Q_savings_2*MJ_kWh)
    b.Q1bigM = Constraint(expr = b.Q_savings_1_gas <= (b.K_boiler[0]+b.K_boiler[2])*10000)
    b.Q2bigM = Constraint(expr = b.Q_savings_2_gas <= (b.K_boiler[0]+b.K_boiler[2])*10000)
    b.Q1elebigM = Constraint(expr = b.Q_savings_1_ele <= (b.K_boiler[1])*10**4)
    b.Q2elebigM = Constraint(expr = b.Q_savings_2_ele <= (b.K_boiler[1])*10**4)
    
    # heat transfer areas for heat exchangers
    b.Area1 = Constraint(expr = b.A_boil == (b.V_water_mash + b.V_water_laut)*Cp_water*(T_store - Tmash)/(LogMeanBoil*4*t_boil))
    b.Area2 = Constraint(expr = b.A_mash == V_wort*Cp_wort*(b.T_post_mash-Tmash)/(LogMeanMash*U_hex*t_mash))
    b.Area3 = Constraint(expr = b.A_laut == V_wort*Cp_wort*(b.T_post_laut-Tlaut)/(LogMeanLaut*U_hex*t_laut))
    
    # total cost of heat storage
    b.HeatStorageCost = Constraint(expr = (((757+3.4*price_insul/CEPCI_2023*CEPCI_2010)*(b.V_water_mash/1000 + b.V_water_laut/1000)+6652*b.K_heat + (1658*b.K_heat +179*b.A_boil + b.K_HEX[0]*(1658)+179*b.A_mash + b.K_HEX[1]*(1658)+179*b.A_laut))/CEPCI_2010*CEPCI_2023) == b.HexCapex)
    #b.K_heat*(11487)+2715*(b.V_water_mash/1000 + b.V_water_laut/1000)-97.4*(b.V_water_mash/1000 + b.V_water_laut/1000)**2)/CEPCI_2010*CEPCI_2023 + 3.4*(b.V_water_mash/1000 + b.V_water_laut/1000)*price_insul
    b.HeatbigM1 = Constraint(expr = b.K_heat*10000 >= b.V_water_mash/1000 + b.V_water_laut/1000)
    b.HeatbigM2 = Constraint(expr = b.K_heat*10000 >= b.A_boil)
    b.HeatbigM3 = Constraint(expr = b.K_HEX[0]*100 >= b.A_mash)
    b.HeatbigM4 = Constraint(expr = b.K_HEX[1]*100 >= b.A_laut)
    b.HeatbigM5 = Constraint(expr = b.K_HEX[0]*10000 >= b.V_water_mash/1000)
    b.HeatbigM6 = Constraint(expr = b.K_HEX[1]*10000 >= b.V_water_laut/1000)
    
    # Variables for anaerobic digestion
    b.K_AD = Var(domain = Binary) # binary variable for implementation of anaerobic digestion
    b.ADcapacity = Var(domain = NonNegativeReals, bounds = (0,ADspace))
    b.K_ADtype = Var(Set(initialize = np.r_[0:2]), domain = Binary) # binary variable determining the type of anaerobic digester (0 = meso, 1 = thermophilic)
    b.ADopex = Var(Set(initialize = np.r_[0:4]), domain = NonNegativeReals) # power required by the immersion heater to maintain AD temperature (for each season)
    b.Tutility = Var(Set(initialize = np.r_[0:4]), domain = NonNegativeReals, bounds = (35, 70)) # temperature of the water
    b.BSGpercent = Var(domain = NonNegativeReals, bounds = (0,1)) # amount of BSG used for digestion
    b.ADenergy = Var(domain = NonNegativeReals) # energy savings from anaerobic digester
    b.ADCapex = Var(domain = NonNegativeReals) # capital costs of anaerobic digester
    
    b.ADcapacitylimit = Constraint(expr = b.BSGpercent*(BSGproduction*(1/BSGdensity + DaysUntilReady*(MesoBiogas*b.K_ADtype[0]+ThermoBiogas*b.K_ADtype[1]))) <= b.ADcapacity)
    b.MethaneProduction = Constraint(expr = b.BSGpercent*BSGproduction*(101325*(MesoCH4*b.K_ADtype[0]+ThermoCH4*b.K_ADtype[1])/(8.3143*(55+273)))*CH4enthalpy >= b.ADenergy)
    b.ADenergybigM = Constraint(expr = b.ADenergy <= b.K_boiler[0]*10**4)
    b.ADenergylimit = Constraint(expr = b.ADenergy*ADDays <= total_heat*365.25)
    b.ADCapital = Constraint(expr = b.ADCapex >= (b.K_AD*(12292)+1733*b.ADcapacity)/CEPCI_2010*CEPCI_2023 + 3.4*b.ADcapacity*price_insul)
    b.ADbigM = Constraint(expr = b.K_AD*10**2 >= b.ADcapacity)
    b.AdtypeCon = Constraint(expr = b.K_ADtype[0] + b.K_ADtype[1] == 1) # AD type can only be mesophilic or thermophilic
    
    def ADSteadyStateHeatBalance(self, k): # calculates opex of AD unit
        return b.ADopex[k] == 3.4*UAD*b.ADcapacity*(Tmeso*b.K_ADtype[0] + Tthermo*b.K_ADtype[1] - To[k])
    b.ADSteadyStateHeatBalance = Constraint(b.k, rule = ADSteadyStateHeatBalance)
    
    # Variables for CCS and DFM
    b.K_CCS = Var(domain = Binary) # binary variable for implementation of the DFM reactor
    b.CCS = Var(domain = NonNegativeReals, bounds = (0, 10)) # capacity of the DFM reactor (metres cubed)
    b.CO2 = Var(domain = NonNegativeReals) # carbon dioxide uptake profile based on the thermal energy usage profile
    b.H2 = Var(domain = NonNegativeReals) # cost of the hydrogen for each hour of using dual function materials
    b.CH4 = Var(domain = NonNegativeReals) # methane produced for each mole
    b.DFMSavings = Var(domain = NonNegativeReals) # energy savings from DFM reactor
    
    b.CO2vol = Var(domain = NonNegativeReals) # volume of CO2 produced
    b.CatalystMass = Var(domain = NonNegativeReals) # mass of catalyst required
    b.DFM_cost = Var(domain = NonNegativeReals) # cost of the DFM reactor
    b.Q_DFM = Var(b.k, b.j, domain = NonNegativeReals) # cost of heating for DFM
    
    YCH4 = 5.8*10**-3*350 - 1.12 # yield of methane at 350oC
    
    def CO2produced(self, j): # kilograms of CO2 produced every hour from combustion
        return b.CO2 >= abs(load[j,0] - load[j,1])*CO2perkWh*b.CCS/maxCap
    b.CO2produced = Constraint(b.j, rule = CO2produced)
    
    b.CO2volume = Constraint(expr = b.CO2vol == b.CO2/44*8.3143*(350+273.15)/101325*10**3) # volume of CO2 produced
    b.CatalystMassRequired = Constraint(expr = b.CatalystMass == b.CO2vol/WHSV/1000) # mass of catalyst required
    b.H2requirement = Constraint(expr = b.CO2/44*2*2 == b.H2) # amount of hydrogen required
    b.DFMsavings = Constraint(expr = b.DFMSavings == b.CO2*YCH4/44*CH4enthalpy) # energy savings from DFM reactor
    b.DFMsavingsbigM = Constraint(expr = b.DFMSavings <= b.K_boiler[0]*10**4)
    b.DFMsize1 = Constraint(expr = b.CCS == b.CO2vol*5) # volume of DFM reactor based on the volume of gas inputted
    b.DFMsize2 = Constraint(expr = b.DFM_cost >= (4147*b.CCS+b.K_CCS*17807)/CEPCI_2010*CEPCI_2023 + 3.4*b.CCS*price_insul) # 
    b.DFMbigM = Constraint(expr = b.K_CCS*10**2 >= b.CCS)
    
    b.x = Set(initialize = np.r_[0:first_index])
    b.y = Set(initialize = np.r_[first_index+1:(first_index+non_zero)])
    b.z = Set(initialize = np.r_[(first_index+non_zero):24])
    
    def DFMopex1(self, i,j):
        return b.Q_DFM[i,j] == 0
    b.DFMopex1 = Constraint(b.k, b.y, rule = DFMopex1)
    
    def DFMopex2(self, i,j):
        return b.Q_DFM[i,j] == U_DFM*3.4*(b.CCS)*(350-To[i])
    b.DFMopex2 = Constraint(b.k, b.y, rule = DFMopex2)
    
    def DFMopex3(self, i,j):
        return b.Q_DFM[i,j] == 0
    b.DFMopex3 = Constraint(b.k, b.y, rule = DFMopex3)
    
    
    a = np.r_[0:24]
    
    # Objective function
    Op_ele_1 = (sum(price_ele*b.p_buy[0,k] for k in a) - sum(sell_ele*b.p_sell[0,k] for k in a) - b.K_boiler[1]*MJ_kWh*price_ele*((b.Q_savings_1 + b.Q_savings_2))) * BrewDaysSpring + (sum(price_ele*b.p_buy[4,k] for k in a) - sum(sell_ele*b.p_sell[4,k] for k in a))*(day[0]-BrewDaysSpring)
    Op_ele_2 = (sum(price_ele*b.p_buy[1,k] for k in a) - sum(sell_ele*b.p_sell[1,k] for k in a) - b.K_boiler[1]*MJ_kWh*price_ele*((b.Q_savings_1 + b.Q_savings_2))) * BrewDaysSummer + (sum(price_ele*b.p_buy[5,k] for k in a) - sum(sell_ele*b.p_sell[5,k] for k in a))*(day[1]-BrewDaysSummer)
    Op_ele_3 = (sum(price_ele*b.p_buy[2,k] for k in a) - sum(sell_ele*b.p_sell[2,k] for k in a) - b.K_boiler[1]*MJ_kWh*price_ele*((b.Q_savings_1 + b.Q_savings_2))) * BrewDaysAutumn + (sum(price_ele*b.p_buy[6,k] for k in a) - sum(sell_ele*b.p_sell[6,k] for k in a))*(day[2]-BrewDaysAutumn)
    Op_ele_4 = (sum(price_ele*b.p_buy[3,k] for k in a) - sum(sell_ele*b.p_sell[3,k] for k in a) - b.K_boiler[1]*MJ_kWh*price_ele*((b.Q_savings_1 + b.Q_savings_2))) * BrewDaysWinter + (sum(price_ele*b.p_buy[7,k] for k in a) - sum(sell_ele*b.p_sell[7,k] for k in a))*(day[3]-BrewDaysWinter)
    Op_heat_1 = ((total_heat-(b.Q_savings_1 + b.Q_savings_2) - b.DFMSavings) * MJ_kWh)  * BrewDaysSpring  # operation cost of gas or oil boilers
    Op_heat_2 = ((total_heat-(b.Q_savings_1 + b.Q_savings_2) - b.DFMSavings) * MJ_kWh)  * BrewDaysSummer  
    Op_heat_3 = ((total_heat-(b.Q_savings_1 + b.Q_savings_2) - b.DFMSavings) * MJ_kWh)  * BrewDaysAutumn  
    Op_heat_4 = ((total_heat-(b.Q_savings_1 + b.Q_savings_2) - b.DFMSavings) * MJ_kWh)  * BrewDaysWinter
    Op_heat = (Op_heat_1+Op_heat_2+Op_heat_3+Op_heat_4 - ADDays*b.ADenergy)*lifetime[f]
    CO2_heat = (Op_heat_1+Op_heat_2+Op_heat_3+Op_heat_4)*lifetime[f]
    b.opex = Objective(expr = ((weight*((Op_ele_1+Op_ele_2+Op_ele_3+Op_ele_4)*lifetime[f] + b.investment + Op_heat*(b.K_boiler[0]*gas_price+b.K_boiler[2]*oil_price)+ b.H2*price_H2*BrewDaysSum*lifetime[f]) + CO2_heat*(CO2oil*b.K_boiler[2]+CO2perkWh*b.K_boiler[0])*tax/1000) ) , sense = minimize) # 
    
    # capital cost constraint, subtracting the investment of the boiler already in place
    if currentBoiler == 0:
        b.investCon = Constraint(expr = b.investment == investment_panel * sum(b.num_solar[k] for k in np.r_[0:max_number]) + b.battery_capacity*investment_battery + investment_boiler_ele*b.K_boiler[1] + investment_boiler_oil*b.K_boiler[2]  + investment_turbine*b.turbine_capacity + (b.CatalystMass*catalystPrice+b.DFM_cost) + b.ADCapex + b.HexCapex)
    elif currentBoiler == 1:
        b.investCon = Constraint(expr = b.investment == investment_panel * sum(b.num_solar[k] for k in np.r_[0:max_number]) + b.battery_capacity*investment_battery + investment_boiler_gas*b.K_boiler[0] + investment_boiler_oil*b.K_boiler[2]  + investment_turbine*b.turbine_capacity + (b.CatalystMass*catalystPrice+b.DFM_cost) + b.ADCapex + b.HexCapex)
    elif currentBoiler == 2:
        b.investCon = Constraint(expr = b.investment == investment_panel * sum(b.num_solar[k] for k in np.r_[0:max_number]) + b.battery_capacity*investment_battery + investment_boiler_gas*b.K_boiler[0] + investment_boiler_ele*b.K_boiler[1]  + investment_turbine*b.turbine_capacity + (b.CatalystMass*catalystPrice+b.DFM_cost) + b.ADCapex + b.HexCapex)
    else:
        print("Assuming no boiler is currently installed")
        b.investCon = Constraint(expr = b.investment == investment_panel * sum(b.num_solar[k] for k in np.r_[0:max_number]) + b.battery_capacity*investment_battery + investment_boiler_gas*b.K_boiler[0] + investment_boiler_ele*b.K_boiler[1] + investment_boiler_oil*b.K_boiler[2]  + investment_turbine*b.turbine_capacity + (b.CatalystMass*catalystPrice+b.DFM_cost) + b.ADCapex)# + b.HexCapex)
    
    def brewingDays(self, k, l):
        return b.p_main[k,l] + b.p_solar[k,l] + b.p_wind[k,l] + b.p_ess[k,l]  == load[l,0]*(b.K_boiler[0]+b.K_boiler[2]) + (load[l,1])*b.K_boiler[1]/0.99 + b.ADopex[k] + b.Q_DFM[k, l]
    b.brewingDays = Constraint(b.k, b.j, rule = brewingDays)
    
    def normalDays(self, k, j):
        return b.p_main[k,j] + b.p_solar[k,j] + b.p_wind[k,j] + b.p_ess[k,j] == 4.4 + b.ADopex[k-4] + b.Q_DFM[k-4, j]
    b.normalDays = Constraint(b.l, b.j, rule = normalDays)
    
    def powerDiff(self, i, j):
        return b.p_main[i,j] == b.p_buy[i,j] - b.p_sell[i,j]
    b.powerDiff = Constraint(b.i,b.j, rule = powerDiff)
    
    def buyTheThing(self,i,j):
        return b.p_buy[i,j] <= b.K_main[i,j] * 10000
    b.buyTheThing = Constraint(b.i, b.j, rule = buyTheThing)
    
    def binder(self, i, j):
        return b.p_sell[i,j] <= (1-b.K_main[i,j])*10000
    b.binder = Constraint(b.i, b.j, rule = binder)
    
    b.t = Set(initialize = np.r_[0:T-1])
    
    def SOCthing1(self, t, k):
        return b.SOC[k,t+1] == b.SOC[k,t] + (b.p_cha[k,t+1]*charge_eff - b.p_dis[k,t+1]/discharge_eff)/nominal_cap
    b.SOCthing1 = Constraint(b.t, b.i, rule = SOCthing1)
    
    def SOCextra(self, k):
        return b.p_dis[k,22] >= b.p_dis[k,23]
    b.SOCextra = Constraint(b.i, rule = SOCextra)
    
    def SOCmax(self, i,j):
        return b.SOC[i,j] <= b.battery_capacity*20.83
    b.SOCmax = Constraint(b.i, b.j, rule = SOCmax)
    
    def SOCthing5(self,i):
        return b.SOC[i,0] == 80/4.8*b.battery_capacity
    b.SOCthing5 = Constraint(b.i, rule = SOCthing5)
    
    def SOCthing4(self,i):
        return b.SOC[i,0] == b.SOC[i,T-1]
    b.SOCthing4 = Constraint(b.i, rule = SOCthing4)
    
    def battery(self,i,j):
        return b.p_ess[i,j] == b.p_dis[i,j] - b.p_cha[i,j]
    b.battery = Constraint(b.i, b.j, rule = battery)
    
    def bigM1(self,i,j):
        return b.p_cha[i,j] <= b.K_ess[i,j] * 10000
    b.bigM1 = Constraint(b.i, b.j, rule = bigM1)
    
    def bigM2(self,i,j):
        return b.p_dis[i,j] <= (1 - b.K_ess[i,j])*10000
    b.bigM2 = Constraint(b.i, b.j, rule = bigM2)
    
    def charger(self,i,j):
        return b.p_cha[i,j] * charge_eff <= b.battery_capacity * 0.5
    b.charger = Constraint(b.i, b.t, rule = charger)
    
    def discharger(self,i,j):
        return b.p_dis[i,j] / discharge_eff <= b.battery_capacity * 0.5
    b.discharger = Constraint(b.i, b.t, rule = discharger)    
    
    def SOCbigM(self,i,j):
        return b.p_ess[i,j] <= b.battery_capacity * 10000
    b.SOCbigM = Constraint(b.i,b.j, rule = SOCbigM)
    
    def PVlimit1(self, i,j):
        return b.p_solar[i,j] <= sum(b.num_solar[f] * P_PV[j,i] for f in np.r_[0:max_number])
    b.PVlimit1 = Constraint(b.k, b.j, rule = PVlimit1)
    
    def PVlimit2(self, i,j):
        return b.p_solar[i,j] <= sum(b.num_solar[f] * P_PV[j,(i-4)] for f in np.r_[0:max_number])
    b.PVlimit2 = Constraint(b.l, b.j, rule = PVlimit2)
    
    def Windlimit1(self, i,j):
        return b.p_wind[i,j] <= windPower[i,j]
    b.Windlimit1 = Constraint(b.k, b.j, rule = Windlimit1)
    
    def Windlimit2(self, i,j):
        return b.p_wind[i,j] <= windPower[i-4,j]
    b.Windlimit2 = Constraint(b.l, b.j, rule = Windlimit2)
    
    def Windlimit3(self, i,j):
        return b.p_wind[i,j] <= b.turbine_capacity
    b.Windlimit3 = Constraint(b.i, b.j, rule = Windlimit3)
    
    b.oneBoiler = Constraint(expr = b.K_boiler[0] + b.K_boiler[1] + b.K_boiler[2] == 1)
    
    solver = SolverFactory('scip', IPOPT = True, WORHP = True, FILTERSQP = True)#
    
    b.turbine_capacity.fix(3)
    b.K_heat.fix(0)
    b.K_AD.fix(0)
    b.K_ADtype[1].fix(1)
    b.K_CCS.fix(0)
    b.K_HEX[0].fix(0)
    b.K_HEX[1].fix(0)
    b.K_boiler[1].fix(1)
    b.battery_capacity.fix(3)
    
    solver.solve(b)
    
    b.battery_capacity.unfix()
    b.K_heat.unfix()
    b.K_HEX[0].unfix()
    b.K_HEX[1].unfix()
    b.K_AD.unfix()
    b.K_ADtype[0].unfix()
    b.turbine_capacity.unfix()
    b.BSGpercent.unfix()
    b.V_water_mash.unfix()
    b.V_water_laut.unfix()
    b.T_post_mash.unfix()
    b.T_post_laut.unfix()
    b.K_boiler[1].unfix()
    b.K_CCS.unfix()
    
    solver.solve(b).write()
    
    
    
    print("Lifetime value...")
    print(lifetime[f])
    print("\n")
    print("Objective value...")
    print(value(b.opex))
    
    print("\n")
    print("Boiler type...")
    print(value(b.K_boiler[0]))
    print(value(b.K_boiler[1]))
    print(value(b.K_boiler[2]))
    print("\n")
    print("Number of panels")
    print(value(sum(b.num_solar[k] for k in np.r_[0:max_number])))
    print(value(investment_panel * sum(b.num_solar[k] for k in np.r_[0:max_number])))
    print("\n")
    print("Turbine capacity")
    print(value(b.turbine_capacity))
    print(value(investment_turbine*b.turbine_capacity))
    print()
    print("\n")
    print("Battery?")
    if round(value(b.battery_capacity)) > 0:
        print("Yes")
        print("\n")
        print("Battery capacity")
        print(value(b.battery_capacity))
        print(value(investment_battery*b.battery_capacity))
        print("\n")
    else:
        print("No")
        print("\n")
    print("Dual function materials?")
    if round(value(b.K_CCS)) == 1 and round(value(b.K_boiler[0])) == 1:
        print("Yes")
        print("\n")
        print("DFM reactor capacity")
        print(value(b.CCS))
        print("\n")
        print("Catalyst mass required")
        print(value(b.CatalystMass))
        print("\n")
    else:
        print("No")
        print("\n")  
    print("Anaerobic digestion?")
    if round(value(b.K_AD)) == 1 and round(value(b.K_boiler[0])) == 1:
        print("Yes")
        print("\n")
        print("AD capacity")
        print(value(b.ADcapacity))
        print(value(b.ADCapex))
        print("\n")
        print("Percentage of BSG used for AD")
        print(value(b.BSGpercent)*100)
        print("\n")
        print("AD type")
        print(value(b.K_ADtype[0]))
        print(value(b.K_ADtype[1]))
        print("\n")
    else:
        print("No")
        print("\n")
    print("Heat storage system?")
    if round(value(b.K_heat)) == 1:
        print("Yes")
        print("\n")
        print("Which heat exchangers?")
        print(value(b.K_HEX[0]))
        print(value(b.K_HEX[1]))
        print("\n")
        print("Heat storage volume")
        print(value(b.V_water_mash*b.K_HEX[0]+b.V_water_laut*b.K_HEX[1]))
        print(value(b.HexCapex))
        print("\n")
        print("Heat exchanger areas")
        print(value(b.A_boil))
        print(value(b.A_mash))
        print(value(b.A_laut))
        print("\n")
    else:
        print("No")
        print("\n")
    #print(value((((1600+3.4*price_insul/CEPCI_2023*CEPCI_2010)*(b.V_water_mash/1000 + b.V_water_laut/1000)**0.7+5800*b.K_heat + (1600*b.K_heat +210*b.A_boil**0.95 + b.K_HEX[0]*(1600)+210*b.A_mash**0.95 + b.K_HEX[1]*(1600)+210*b.A_laut**0.95))/CEPCI_2010*CEPCI_2023)))
    #print(value(((1600+3.4*price_insul/CEPCI_2023*CEPCI_2010)*(b.V_water_mash/1000 + b.V_water_laut/1000)**0.7+5800*b.K_heat)/CEPCI_2010*CEPCI_2023))
    #print(value((1600*b.K_heat +210*b.A_boil**0.95)/CEPCI_2010*CEPCI_2023))
    #print(value((b.K_HEX[0]*(1600)+210*b.A_mash**0.95)/CEPCI_2010*CEPCI_2023))
    #print(value((b.K_HEX[1]*(1600)+210*b.A_laut**0.95)/CEPCI_2010*CEPCI_2023))
    #print(value(Op_heat*(b.K_boiler[0]*gas_price+b.K_boiler[2]*oil_price))/lifetime[f])
    #print(value(CO2_heat*(CO2oil*b.K_boiler[2]+CO2perkWh*b.K_boiler[0])/1000) )
    #print(value(b.Q_savings_1+b.Q_savings_2)*BrewDaysSum)
    #print(value(ADDays*b.ADenergy))
    #print(value(Op_ele_1+Op_ele_2+Op_ele_3+Op_ele_4))
