#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decode RC+S JSON file and save as MAT file.

Converting all Enumeration to Readable Text based on Summit 1.6.0 API Documentation

@author: Jackson Cagle, 2020
"""

######################################################################
######################### Importing Libraries ########################
######################################################################
import json
import scipy.io as sio
import sys
import os
import copy
import numpy as np
from datetime import datetime

from utility.PythonUtility import unwrap

######################################################################
#################### Summit API 1.6.0 Defines ########################
######################################################################
AdaptiveTherapyModes = {0: "Disabled", 1: "Operative", 2: "Embedded"}
AdaptiveTherapyStatus = {0: "Inactive", 1: "OperativeActive", 2: "EmbeddedActive"}
AdaptiveTherapyState = {0: "State0", 1: "State1", 2: "State2", 3: "State3", 4: "State4",
                        5: "State5", 6: "State6", 7: "State7", 8: "State8", 15: "NoState"}

GroupNumber = {0: "GroupA", 16: "GroupB", 32: "GroupC", 48: "GroupD"}
AmplitudeResolution = {0: 0.1, 1: 0.2} # in mA. 
CyclingUnits = {0: 0.1, 1: 1, 2: 10} # in seconds. This is the scaling factor
TherapyStatus = {0: "Off", 1: "On", 2: "LeadIntegrity", 3: "TransitionToOff", 4: "TransitionToActive", 5: "TransitionToLeadIntegrity"}
RampTypes = {0: "None", 1: "UpEnabled", 2: "DownEnabled", 4: "RepeatRampUp"}
ElectrodeType = {0: "-", 1: "+"} # 0x00 = Cathode, 0x01 = Anode

AccelSampleRate = {0: 64, 1: 32, 2: 16, 3: 8, 4: 4, 255: 0}
SenseStates = {0: "None", 1: "Lfp", 2: "FFT", 4: "Power", 8: "Unused08", 16: "DetectionLd0", 32: "DetectionLd1", 64: "LoopRecording", 128: "AdaptiveStim"}
SenseValids = {0: "None", 1: "Lfp", 2: "FFT", 4: "Power", 8: "Misc", 16: "DetectionLd0", 32: "DetectionLd1"}

BlankRegisterValues = {0: "None", 1: "Stim1RisingEdge", 2: "Stim1FallingEdge", 4: "Stim2RisingEdge", 8: "Stim2FallingEdge", 16: "ActiveRechargeRisingEdge", 32: "ActiveRechargeFallingEdge", 64: "PassiveRechargeRisingEdge", 128: "PassiveRechargeFallingEdge"}

TdChannelEnable = {0: "Enable", 15: "Disable"}
TdEvokedResponseEnable = {0: "Standard", 16: "Evoked 0 Input", 32: "Evoked 1 Input"}
TdGains = {0: 500, 1: 1000, 2: 250, 4: 2000}
TdHpfs = {0: "0.85Hz", 16: "1.2Hz", 32: "3.3Hz", 96: "8.6Hz"}
TdLpfStage1 = {9: "450Hz", 18: "100Hz", 36: "50Hz"}
TdLpfStage2 = {9: "100Hz", 11: "160Hz", 12: "350Hz", 14: "1700Hz"}
TdLpfStage2CurrentModes = {1: "Current 2.0x", 2: "Current 0.5x"}
TdLpfStage2Outputs = {0: "Common VCM", 1: "Common Mode"}
TdMuxInputs = {0: "Floating", 1: "E00", 2: "E01", 4: "E02", 8: "E03", 16: "E04", 32: "E05", 64: "E06", 128: "E07"}
TdSampleRates = {0: 250, 1: 500, 2: 1000, 240: 0}

FftWeightMultiplies = {8: "Shift7", 9: "Shift6", 10: "Shift5", 11: "Shift4", 12: "Shift3", 13: "Shift2", 14: "Shift1", 15: "Shift0"}
FftSizes = {0: 64, 1: 256, 3: 1024}
FftWindowAutoLoads = {2: "Hann100", 22: "Hann50", 42: "Hann25"}

PowerBandEnables = {1: "Ch0Band0", 2: "Ch0Band1", 4: "Ch1Band0", 8: "Ch1Band1", 16: "Ch2Band0", 32: "Ch2Band1", 64: "Ch3Band0", 128: "Ch3Band1"}
BridgingConfig = {0: "None", 4: "Bridge 0 to 2", 8: "Bridge 1 to 3"}
LoopRecordingTriggers = {0: "None", 1: "State0", 2: "State1", 4: "State2", 8: "State3", 16: "State4", 32: "State5", 64: "State6", 128: "State7", 256: "State8"}

SenseTimeDomainDebugInfo = {0: "None", 1: "Ch0 Overlow", 2: "Ch1 Overlow", 4: "Ch2 Overlow", 8: "Ch3 Overlow", 64: "HWLP Wrapped Packet", 128: "Sensing Lost Interupt"}

EventIDs = {
    0: "AdaptiveTherapyStateChange",
    1: "AdaptiveTherapyStateWritten",
    2: "LfpLoopRecorderEvent",
    4: "LfpSenseStateEvent",
    5: "LdDetectionEvent",
    16: "Cycling",
    17: "CpSession",
    18: "TherapyStatus",
    20: "TherapyAvailability",
    22: "NonSessionRecharge",
    24: "RechargeSesson",
    26: "ActiveDeviceChanged",
    27: "AdaptiveTherapyStatusChanged",
    255: "Invalid"
}


#--------------------------------------------------------------------------
# Copyright (c) Medtronic, Inc. 2017
#
# MEDTRONIC CONFIDENTIAL -- This document is the property of Medtronic
# PLC, and must be accounted for. Information herein is confidential trade
# secret information. Do not reproduce it, reveal it to unauthorized 
# persons, or send it outside Medtronic without proper authorization.
#--------------------------------------------------------------------------
#
# Authors: Ben Johnson (johnsb68), Jackson Cagle
#
# Description: This function fixes a malformed Summit JSON file, which arises
# when a SummitSystem session is not closed properly.
#
# -------------------------------------------------------------------------
def fixMalformedJson(jsonString, fileType):
    jsonString = jsonString.replace("INF","Inf")
    
    numOpenSqua = jsonString.count("[")
    numOpenCurl = jsonString.count("{")
    numCloseCurl = jsonString.count("}")
    numCloseSqua = jsonString.count("]")
    
    if numOpenSqua is not numCloseSqua and (fileType.find("Log") >= 0 or fileType.find("Settings") >= 0):
        jsonStringOut = jsonString + "]"
        #print(f"Your {fileType}.json file appears to be malformed, a fix was attempted in order to proceed with processing")
    elif numOpenSqua is not numCloseSqua or numOpenCurl is not numCloseCurl:
        jsonStringfix = "}"*(numOpenCurl-numCloseCurl-1) + "]"*(numOpenSqua-numCloseSqua-1) + "}]"
        jsonStringOut = jsonString + jsonStringfix
        #print(f"Your {fileType}.json file appears to be malformed, a fix was attempted in order to proceed with processing")
    else:
        jsonStringOut = jsonString
    
    return jsonStringOut

def decodeJSON(inputFilename, fileType="DeviceSettings"):
    try:
        Data = json.load(open(inputFilename))
        
    except json.JSONDecodeError:
        fid = open(inputFilename, "r")
        jsonString = fid.read()
        fid.close()
        
        jsonStringOut = fixMalformedJson(jsonString, fileType)
        Data = json.loads(jsonStringOut)
        
    return Data

def getDeviceSettings(DataFolder):
    JSON = decodeJSON(DataFolder + "DeviceSettings.json", fileType="DeviceSettings")
    DeviceConfigurations = list()
    for DeviceConfiguration in JSON:
        
        # Setup Initial Dictionary Variable
        Configuration = dict()
        
        # Telemetry Info (When is the following configuration transmitted)
        if "RecordInfo" in DeviceConfiguration.keys():
            Configuration["Time"] = DeviceConfiguration["RecordInfo"]["HostUnixTime"] / 1000.0
            # TODO: Add UTC Offset
            Configuration["SessionDateTime"] = datetime.fromtimestamp(float(DeviceConfiguration["RecordInfo"]["SessionId"]) / 1000.0) 
        
        # General Setup
        if "GeneralData" in DeviceConfiguration.keys():
            Configuration["GeneralData"] = DeviceConfiguration["GeneralData"]
        
        # Telemetry Reconnection?
        if "TelemetryModuleInfo" in DeviceConfiguration.keys():
            Configuration["TelemetryModule"] = DeviceConfiguration["TelemetryModuleInfo"]
        
        # Adaptive Configuration
        if "AdaptiveConfig" in DeviceConfiguration.keys():
            Configuration["Adaptive"] = dict()
            if type(DeviceConfiguration["AdaptiveConfig"]) is dict:
                
                for key in DeviceConfiguration["AdaptiveConfig"].keys():
                    if key == "adaptiveMode":
                        Configuration["Adaptive"]["Mode"] = AdaptiveTherapyModes[DeviceConfiguration["AdaptiveConfig"]["adaptiveMode"]]
                    
                    if key == "adaptiveStatus":
                        Configuration["Adaptive"]["Status"] = AdaptiveTherapyStatus[DeviceConfiguration["AdaptiveConfig"]["adaptiveStatus"]]
                    
                    if key == "currentState":
                        Configuration["Adaptive"]["CurrentState"] = AdaptiveTherapyState[DeviceConfiguration["AdaptiveConfig"]["currentState"]]
                    
                    if key == "deltaLimitsValid":
                        Configuration["Adaptive"]["DeltaLimitsValid"] = DeviceConfiguration["AdaptiveConfig"]["deltaLimitsValid"]
                    
                    if key == "deltaUpperLimits":
                        Configuration["Adaptive"]["DeltaLimits"] = DeviceConfiguration["AdaptiveConfig"]["deltaUpperLimits"]
                    
                    if key == "deltasValid":
                        Configuration["Adaptive"]["DeltasValid"] = DeviceConfiguration["AdaptiveConfig"]["deltasValid"]
                        Configuration["Adaptive"]["Delta"] = DeviceConfiguration["AdaptiveConfig"]["deltas"]
                    
                    if key == "initialState":
                        Configuration["Adaptive"]["InitialState"] = DeviceConfiguration["AdaptiveConfig"]["initialState"]
                        Configuration["Adaptive"]["State"] = list()
                        for n in range(9):
                            Configuration["Adaptive"]["State"].append(DeviceConfiguration["AdaptiveConfig"]["state" + str(n)])

        # Adaptive Configuration
        if "DetectionConfig" in DeviceConfiguration.keys():
            Configuration["Detector"] = DeviceConfiguration["DetectionConfig"]
                
        # Therapy Configuration
        for n in range(4):
            if "TherapyConfigGroup" + str(n) in DeviceConfiguration.keys():
                if not "Therapy" in Configuration:
                    Configuration["Therapy"] = list()
                
                TherapyConfig = dict()
                if DeviceConfiguration["TherapyConfigGroup" + str(n)]["valid"]:
                    TherapyConfig["Valid"] = True
                    TherapyConfig["GroupNumber"] = GroupNumber[DeviceConfiguration["TherapyConfigGroup" + str(n)]["Index"]]                
                    TherapyConfig["AmplitudeResolution"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["amplitudeResolution0_2mA"]
                    # As of Summit 1.6.0, the Amplitude Resolution is identical to the Group Number Index, possible bug in the firmware.                
                    #TherapyConfig["AmplitudeResolution"] = AmplitudeResolution[DeviceConfiguration["TherapyConfigGroup" + str(n)]["amplitudeResolution0_2mA"]]
                    TherapyConfig["Cycling"] = dict()
                    TherapyConfig["Cycling"]["Enabled"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["cyclingEnabled"]
                    TherapyConfig["Cycling"]["OnTime"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["cycleOnTime"]["time"] * CyclingUnits[DeviceConfiguration["TherapyConfigGroup" + str(n)]["cycleOnTime"]["units"]]
                    TherapyConfig["Cycling"]["OffTime"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["cycleOffTime"]["time"] * CyclingUnits[DeviceConfiguration["TherapyConfigGroup" + str(n)]["cycleOffTime"]["units"]]
                    TherapyConfig["Mode"] = TherapyStatus[DeviceConfiguration["TherapyConfigGroup" + str(n)]["mode"]]
                    TherapyConfig["PulseWidthLimits"] = (DeviceConfiguration["TherapyConfigGroup" + str(n)]["pulseWidthLowerLimitInMicroseconds"],
                                                         DeviceConfiguration["TherapyConfigGroup" + str(n)]["pulseWidthUpperLimitInMicroseconds"])
                    if DeviceConfiguration["TherapyConfigGroup" + str(n)]["rampRepeat"] == 0:
                        TherapyConfig["RampType"] = "None"
                    else:
                        TherapyConfig["RampType"] = list()
                        for i in range(4):
                            if np.bitwise_and(DeviceConfiguration["TherapyConfigGroup" + str(n)]["rampRepeat"] >> i, 1) > 0:
                                TherapyConfig["RampType"].append(RampTypes[1 << i])
                    TherapyConfig["RampTime"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["rampTime"] * 0.1 # in seconds
                    TherapyConfig["FrequencyLimits"] = (DeviceConfiguration["TherapyConfigGroup" + str(n)]["rateLowerLimitInHz"],
                                                         DeviceConfiguration["TherapyConfigGroup" + str(n)]["rateUpperLimitInHz"])
                    TherapyConfig["Frequency"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["rateInHz"]
                    
                    TherapyConfig["Programs"] = list()
                    for i in range(4):
                        ProgramSetting = dict()
                        ProgramSetting["Valid"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["valid"]
                        if ProgramSetting["Valid"]:
                            ProgramSetting["Enabled"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["isEnabled"]
                            ProgramSetting["Amplitude"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["amplitudeInMilliamps"]
                            if "AmplitudeLimitsGroup" + str(n) in DeviceConfiguration.keys():
                                ProgramSetting["AmplitudeLimits"] = (DeviceConfiguration["AmplitudeLimitsGroup" + str(n)]["prog" + str(i) + "LowerInMilliamps"],
                                                                     DeviceConfiguration["AmplitudeLimitsGroup" + str(n)]["prog" + str(i) + "UpperInMilliamps"])
                            ProgramSetting["PulseWidth"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["pulseWidthInMicroseconds"]
                            ProgramSetting["ActiveRechargeRatio"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["miscSettings"]["activeRechargeRatio"] / 10.0
                            ProgramSetting["Cycling"] = DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["miscSettings"]["cyclingEnable"]
                            ProgramSetting["Electrode"] = ""
                            ProgramSetting["ElectrodeList"] = list()
                            for e in range(16):
                                if not DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][e]["isOff"]:
                                    ProgramSetting["Electrode"] += ElectrodeType[DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][e]["electrodeType"]] + str(e)
                                    ProgramSetting["ElectrodeList"].append({
                                        "ElectrodeState": ElectrodeType[DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][e]["electrodeType"]],
                                        "Channel": e
                                    })
                            if not DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][16]["isOff"]:
                                ProgramSetting["Electrode"] += ElectrodeType[DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][16]["electrodeType"]] + "CAN"
                                ProgramSetting["ElectrodeList"].append({
                                    "ElectrodeState": ElectrodeType[DeviceConfiguration["TherapyConfigGroup" + str(n)]["programs"][i]["electrodes"]["electrodes"][16]["electrodeType"]],
                                    "Channel": "CAN"
                                })
                        TherapyConfig["Programs"].append(ProgramSetting)
                else:
                    TherapyConfig["Valid"] = False
                Configuration["Therapy"].append(TherapyConfig)
        
        # Battery Status
        if "BatteryStatus" in DeviceConfiguration.keys():
            Configuration["BatteryStatus"] = dict()
            Configuration["BatteryStatus"]["BatteryLevel"] = DeviceConfiguration["BatteryStatus"]["batteryLevelPercent"]
            Configuration["BatteryStatus"]["CurrentSoC"] = DeviceConfiguration["BatteryStatus"]["batterySOC"] # mAh unit
            Configuration["BatteryStatus"]["FailureSoC"] = DeviceConfiguration["BatteryStatus"]["therapyUnavailableSOC"] # mAh unit
            Configuration["BatteryStatus"]["Voltage"] = DeviceConfiguration["BatteryStatus"]["batteryVoltage"] / 1000.0 # Volt unit. 0xFFFF is invalid
            Configuration["BatteryStatus"]["EstimatedCapacity"] = DeviceConfiguration["BatteryStatus"]["estimatedCapacity"] # mAh unit
            Configuration["BatteryStatus"]["ManufacturedCapacity"] = DeviceConfiguration["BatteryStatus"]["manufacturedCapacity"] # mAh unit
            if "GeneralData" in DeviceConfiguration.keys():
                Configuration["BatteryStatus"]["DaysUntilEOS"] = DeviceConfiguration["GeneralData"]["daysUntilEos"]
        
        # TODO: OOR Status 
        
        # Current Sensing State
        if "SenseState" in DeviceConfiguration.keys():
            Configuration["SenseStates"] = dict()
            if "accelRate" in DeviceConfiguration["SenseState"].keys():
                Configuration["SenseStates"]["Accelerometer"] = AccelSampleRate[DeviceConfiguration["SenseState"]["accelRate"]]
                
            if "fftStreamChannel" in DeviceConfiguration["SenseState"].keys():
                Configuration["SenseStates"]["FFTChannel"] = DeviceConfiguration["SenseState"]["fftStreamChannel"]
                
            if "state" in DeviceConfiguration["SenseState"].keys():
                if DeviceConfiguration["SenseState"]["state"] == 0:
                    Configuration["SenseStates"]["SenseState"] = "None"
                else:
                    Configuration["SenseStates"]["SenseState"] = list()
                    for n in range(8):
                        if np.bitwise_and(DeviceConfiguration["SenseState"]["state"] >> n, 1) > 0:
                            Configuration["SenseStates"]["SenseState"].append(SenseStates[1 << n])

        # Current Streaming State
        if "StreamState" in DeviceConfiguration.keys():
            if DeviceConfiguration["StreamState"]["StreamsEnabled"]:
                Configuration["StreamState"] = list()
                if DeviceConfiguration["StreamState"]["AccelStreamEnabled"]:
                    Configuration["StreamState"].append("Accelerometer")
                    
                if DeviceConfiguration["StreamState"]["AdaptiveStreamEnabled"]:
                    Configuration["StreamState"].append("AdaptiveStim")
                    
                if DeviceConfiguration["StreamState"]["DetectionStreamEnabled"]:
                    Configuration["StreamState"].append("DetectionLd")
                    
                if DeviceConfiguration["StreamState"]["FftStreamEnabled"]:
                    Configuration["StreamState"].append("FFT")
                    
                if DeviceConfiguration["StreamState"]["LoopRecordMarkerEchoStreamEnabled"]:
                    Configuration["StreamState"].append("LoopRecording")
                    
                if DeviceConfiguration["StreamState"]["PowerDomainStreamEnabled"]:
                    Configuration["StreamState"].append("Power")
                    
                if DeviceConfiguration["StreamState"]["TimeDomainStreamEnabled"]:
                    Configuration["StreamState"].append("Lfp")
                    
                if DeviceConfiguration["StreamState"]["TimeSyncStreamEnabled"]:
                    Configuration["StreamState"].append("TimeSync")
            else:
                Configuration["StreamState"] = "None"
            
        # Reads the lead target and diagnosis type from a raw device settings
        # file. Those recordings are withheld from this deposit, and the shipped
        # HDF5 extracts carry only spectral and pain arrays, so this branch is
        # unreachable for the deposited data.
        if "SubjectInfo" in DeviceConfiguration.keys():
            Configuration["LeadLocation"] = DeviceConfiguration["SubjectInfo"]["LeadTargets"]
            Configuration["DiagnosisType"] = DeviceConfiguration["SubjectInfo"]["Diagnosis"]
            
        # Sensing Configurations
        if "SensingConfig" in DeviceConfiguration.keys():
            Configuration["SensingConfiguration"] = dict()
            if "Valids" in DeviceConfiguration["SensingConfig"]:
                if DeviceConfiguration["SensingConfig"]["Valids"] == 0:
                    Configuration["SensingConfiguration"]["ValidConfiguration"] = "None"
                else:
                    Configuration["SensingConfiguration"]["ValidConfiguration"] = list()
                    for n in range(6):
                        if np.bitwise_and(DeviceConfiguration["SensingConfig"]["Valids"] >> n, 1) > 0:
                            Configuration["SensingConfiguration"]["ValidConfiguration"].append(SenseValids[1 << n])
            
            if "senseBlanking" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["SenseBlanking"] = dict()
                Configuration["SensingConfiguration"]["SenseBlanking"]["Mode"] = BlankRegisterValues[DeviceConfiguration["SensingConfig"]["senseBlanking"]["l380BlankEnableRegisterMode"]]
                Configuration["SensingConfiguration"]["SenseBlanking"]["Duration"] = DeviceConfiguration["SensingConfig"]["senseBlanking"]["blankingExtensionTime"] * 10 + 5 # microseconds
                if Configuration["SensingConfiguration"]["SenseBlanking"]["Duration"] < 25:
                    Configuration["SensingConfiguration"]["SenseBlanking"]["Duration"] = 25
                
            if "timeDomainChannels" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["Lfp"] = list()
                for n in range(4):
                    TimeDomainChannel = dict()
                    TimeDomainChannel["Gain"] = TdGains[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["gain"]]
                    TimeDomainChannel["SamplingRate"] = TdSampleRates[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["sampleRate"]]
                    TimeDomainChannel["Channels"] = (TdMuxInputs[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["plusInput"]],
                                                     TdMuxInputs[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["minusInput"]])
                    TimeDomainChannel["Hpf"] = TdHpfs[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["hpf"]]
                    TimeDomainChannel["Lpf1"] = TdLpfStage1[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["lpf1"]]
                    TimeDomainChannel["Lpf2"] = TdLpfStage2[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["lpf2"]]
                    TimeDomainChannel["Lpf2Mode"] = TdLpfStage2CurrentModes[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["currentMode"]]
                    TimeDomainChannel["Lpf2Output"] = TdLpfStage2Outputs[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["outputMode"]]
                    TimeDomainChannel["EvokedMode"] = TdEvokedResponseEnable[DeviceConfiguration["SensingConfig"]["timeDomainChannels"][n]["evokedMode"]]
                    Configuration["SensingConfiguration"]["Lfp"].append(TimeDomainChannel)
                    
            if "fftConfig" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["FFT"] = dict()
                Configuration["SensingConfiguration"]["FFT"]["WeightMultipliers"] = FftWeightMultiplies[DeviceConfiguration["SensingConfig"]["fftConfig"]["bandFormationConfig"]]
                Configuration["SensingConfiguration"]["FFT"]["Interval"] = DeviceConfiguration["SensingConfig"]["fftConfig"]["interval"] # ms unit
                Configuration["SensingConfiguration"]["FFT"]["NFFT"] = FftSizes[DeviceConfiguration["SensingConfig"]["fftConfig"]["size"]] # this is nFFT
                Configuration["SensingConfiguration"]["FFT"]["StreamOffset"] = DeviceConfiguration["SensingConfig"]["fftConfig"]["streamOffsetBins"] 
                if DeviceConfiguration["SensingConfig"]["fftConfig"]["streamSizeBins"] == 0:
                    Configuration["SensingConfiguration"]["FFT"]["StreamBinSize"] = Configuration["SensingConfiguration"]["FFT"]["NFFT"] / 2
                else:
                    Configuration["SensingConfiguration"]["FFT"]["StreamBinSize"] = DeviceConfiguration["SensingConfig"]["fftConfig"]["streamSizeBins"]
                Configuration["SensingConfiguration"]["FFT"]["WindowType"] = FftWindowAutoLoads[DeviceConfiguration["SensingConfig"]["fftConfig"]["windowLoad"]]
                Configuration["SensingConfiguration"]["FFT"]["Reserved"] = DeviceConfiguration["SensingConfig"]["fftConfig"]["config"] # This is reserved config byte, not used
            
            if "bandEnable" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["PowerEnable"] = list()
                for n in range(8):
                    if np.bitwise_and(DeviceConfiguration["SensingConfig"]["bandEnable"] >> n, 1) > 0:
                        Configuration["SensingConfiguration"]["PowerEnable"].append(PowerBandEnables[1 << n])
            
            if "powerChannels" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["Power"] = np.zeros((8,2))
                for n in range(4):
                    Configuration["SensingConfiguration"]["Power"][n*2+0, 0] = DeviceConfiguration["SensingConfig"]["powerChannels"][n]["band0Start"]
                    Configuration["SensingConfiguration"]["Power"][n*2+1, 0] = DeviceConfiguration["SensingConfig"]["powerChannels"][n]["band1Start"]
                    Configuration["SensingConfiguration"]["Power"][n*2+0, 1] = DeviceConfiguration["SensingConfig"]["powerChannels"][n]["band0Stop"]
                    Configuration["SensingConfiguration"]["Power"][n*2+1, 1] = DeviceConfiguration["SensingConfig"]["powerChannels"][n]["band1Stop"]
        
            if "miscSensing" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["Misc"] = dict()
                Configuration["SensingConfiguration"]["Misc"]["Bridging"] = BridgingConfig[DeviceConfiguration["SensingConfig"]["miscSensing"]["bridging"]]
                Configuration["SensingConfiguration"]["Misc"]["StreamRate"] = DeviceConfiguration["SensingConfig"]["miscSensing"]["streamingRate"] * 10 # ms
                Configuration["SensingConfiguration"]["Misc"]["LoopRecord"] = dict()
                Configuration["SensingConfiguration"]["Misc"]["LoopRecord"]["Trigger"] = list()
                for n in range(9):
                    if np.bitwise_and(DeviceConfiguration["SensingConfig"]["miscSensing"]["lrTriggers"] >> n, 1) > 0:
                        Configuration["SensingConfiguration"]["Misc"]["LoopRecord"]["Trigger"].append(LoopRecordingTriggers[1 << n])
                Configuration["SensingConfiguration"]["Misc"]["LoopRecord"]["PostBuffer"] = DeviceConfiguration["SensingConfig"]["miscSensing"]["lrPostBufferTime"]
            
            if "chopClockSettings" in DeviceConfiguration["SensingConfig"].keys():
                Configuration["SensingConfiguration"]["ChopClockSettings"] = np.zeros((12,1))
                for n in range(12):
                    Configuration["SensingConfiguration"]["ChopClockSettings"][n] = 1000000 / ((np.bitwise_and(DeviceConfiguration["SensingConfig"]["chopClockSettings"]["ChopClockSettings"][n],15) + 1) * 20)
                
        DeviceConfigurations.append(Configuration)
    return DeviceConfigurations


def getTimeDomainData(DataFolder):
    JSON = decodeJSON(DataFolder + "RawDataTD.json", fileType="RawDataTD")
    TimeDomainDatas = list()
    for TimeDomainData in JSON:
        Data = dict()
        if "TimeDomainData" in TimeDomainData.keys():
            if TimeDomainData["TimeDomainData"] != []:
                Data["SystemTick"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["Timestamp"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["UnixTime"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["PacketGenTime"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["Sequences"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["Channels"] = np.zeros((len(TimeDomainData["TimeDomainData"]),4))
                Data["Size"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["SamplingRate"] = np.zeros((len(TimeDomainData["TimeDomainData"]),1))
                Data["Data"] = [None]*len(TimeDomainData["TimeDomainData"])
                Data["EvokedMarker"] = [None]*len(TimeDomainData["TimeDomainData"])
                Data["DebugInfo"] = [None]*len(TimeDomainData["TimeDomainData"])
                for PacketID in range(len(TimeDomainData["TimeDomainData"])):
                    Data["SystemTick"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["Header"]["systemTick"]
                    Data["Timestamp"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["Header"]["timestamp"]["seconds"]
                    Data["UnixTime"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["PacketRxUnixTime"]
                    Data["PacketGenTime"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["PacketGenTime"]
                    Data["Sequences"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["Header"]["dataTypeSequence"]
                    
                    for channel in range(4):
                        if np.bitwise_and(TimeDomainData["TimeDomainData"][PacketID]["IncludedChannels"] >> channel, 1) > 0:
                            Data["Channels"][PacketID,channel] = 1
                    Data["Size"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["Header"]["dataSize"] / 2 / np.sum(Data["Channels"][PacketID,:])
                    Data["SamplingRate"][PacketID] = TdSampleRates[TimeDomainData["TimeDomainData"][PacketID]["SampleRate"]]
                    
                    Data["Size"][PacketID] = len(TimeDomainData["TimeDomainData"][PacketID]["ChannelSamples"][0]["Value"])
                    Data["Data"][PacketID] = np.zeros((int(Data["Size"][PacketID]),4))

                    for tdPacket in TimeDomainData["TimeDomainData"][PacketID]["ChannelSamples"]:
                        Data["Data"][PacketID][:,tdPacket["Key"]] = tdPacket["Value"]
                    
                    Data["EvokedMarker"][PacketID] = TimeDomainData["TimeDomainData"][PacketID]["EvokedMarker"]
                    
                    # Debug Measurement for Overflows
                    if TimeDomainData["TimeDomainData"][PacketID]["DebugInfo"] == 0: 
                        Data["DebugInfo"][PacketID] = "None"
                    else:
                        Data["DebugInfo"][PacketID] = list()
                        for n in range(8):
                            if np.bitwise_and(TimeDomainData["TimeDomainData"][PacketID]["DebugInfo"] >> n, 1) > 0:
                                Data["DebugInfo"][PacketID].append(SenseTimeDomainDebugInfo[1 << n])
    
    
                # Correct for Missing Timestamps
                Data["SystemTick"] = unwrap(Data["SystemTick"], cap=65536).flatten() * 0.0001
                Data["Timestamp"] = Data["Timestamp"].flatten()
                Data["Sequences"] = unwrap(Data["Sequences"], cap=256).flatten()
                Data["UnixTime"] = Data["UnixTime"].flatten() / 1000
                Data["PacketGenTime"] = Data["PacketGenTime"].flatten() / 1000
                Data["Size"] = Data["Size"].flatten()
        TimeDomainDatas.append(Data)
        
    return TimeDomainDatas

def fixMissingPacket_TD(Data, StreamRate, PacketSize):
    FixedData = copy.deepcopy(Data)
    FixedData["Sequences"] = FixedData["Sequences"].astype(np.float64)
    
    # Adjust SystemTicks based on Timestamp
    for PacketID in range(1, len(FixedData["Timestamp"])):
        if FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID - 1] > 6:
            SecondsDifference = FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID-1]
            skippedLaps = np.round((SecondsDifference - (FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1])) / 6.5536)
            FixedData["SystemTick"][PacketID:] += skippedLaps * 6.5536
            SequenceJump = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1]) / StreamRate)
            FixedData["Sequences"][PacketID:] += (SequenceJump - (FixedData["Sequences"][PacketID] - FixedData["Sequences"][PacketID-1]))

    # Remove Bad Packets    
    SelectedPackets = np.concatenate(([0], np.where(np.diff(FixedData["SystemTick"]) > 0)[0].astype(int) + 1))
    DroppedPackets = np.where(np.diff(FixedData["SystemTick"]) <= 0)[0].astype(int) + 1
    adjustedIndex = 0
    for PacketID in DroppedPackets:
        FixedData["Data"].pop(PacketID - adjustedIndex)
        FixedData["EvokedMarker"].pop(PacketID - adjustedIndex)
        FixedData["DebugInfo"].pop(PacketID - adjustedIndex)
        adjustedIndex += 1
        
    for field in FixedData.keys():
        if field == "Channels":
            FixedData[field] = FixedData[field][SelectedPackets,:]
        elif field != "Data" and field != "EvokedMarker" and field != "DebugInfo" and field != "Configuration":
            FixedData[field] = FixedData[field][SelectedPackets]
    
    # Calculating the Drift Coefficient
    FixedData["SystemTick"] -= FixedData["SystemTick"][0]
    UnixTimer = FixedData["UnixTime"] - FixedData["UnixTime"][0]
    coefficients = np.polynomial.polynomial.polyfit(UnixTimer, FixedData["SystemTick"] - UnixTimer, 1)
    FixedData["DriftSlope"] = 1 / (1 + coefficients[1]);
    
    # Use the SystemTicks as adjustment for missing packets
    PacketID =  len(FixedData["SystemTick"])
    while PacketID < len(FixedData["SystemTick"]):
        skipSequence = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID - 1]) / StreamRate).astype(int)
        additionalPackets = np.round(FixedData["Size"][PacketID] / PacketSize).astype(int)
        skipSequence -= (additionalPackets - 1)
        if skipSequence == 0: 
            skipSequence = 1
        if skipSequence > 1:
            FixedData["SystemTick"] = np.insert(FixedData["SystemTick"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["UnixTime"] = np.insert(FixedData["UnixTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["PacketGenTime"] = np.insert(FixedData["PacketGenTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Timestamp"] = np.insert(FixedData["Timestamp"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Size"] = np.insert(FixedData["Size"], PacketID, PacketSize * np.ones(skipSequence - 1))
            FixedData["Channels"] = np.insert(FixedData["Channels"], PacketID, 0 * np.ones((skipSequence - 1, 4)), axis=0)
            FixedData["SamplingRate"] = np.insert(FixedData["SamplingRate"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Sequences"] = np.insert(FixedData["Sequences"], PacketID, -1 * np.ones(skipSequence - 1))
            for i in range(skipSequence-1):
                FixedData["Data"].insert(int(PacketID), np.zeros((int(PacketSize),1)))
                FixedData["EvokedMarker"].insert(int(PacketID), [None])
                FixedData["DebugInfo"].insert(int(PacketID), [None])
        PacketID += skipSequence
    
    # Extract the LFP data to ndarray
    FixedData["LFP"] = np.zeros((int(np.sum(FixedData["Size"])), 4))
    FixedData["Missing"] = np.zeros((int(np.sum(FixedData["Size"])), 1))
    for n in range(len(FixedData["Data"])):
        if FixedData["SystemTick"][n] == -1:
            FixedData["Missing"][int(np.sum(FixedData["Size"][:n])) : int(np.sum(FixedData["Size"][:n+1])),:] = 1
        FixedData["LFP"][int(np.sum(FixedData["Size"][:n])) : int(np.sum(FixedData["Size"][:n+1])),:] = FixedData["Data"][n]
    FixedData["Time"] = np.array(range(int(np.sum(FixedData["Size"])))) / FixedData["SamplingRate"][0] * FixedData["DriftSlope"]
    
    return FixedData
     
def getFFTData(DataFolder):
    JSON = decodeJSON(DataFolder + "RawDataFFT.json", fileType="RawDataFFT")
    FFTDatas = list()
    for FFTData in JSON:
        Data = dict()
        if "FftData" in FFTData.keys():
            if FFTData["FftData"] != []:
                Data["SystemTick"] = np.zeros((len(FFTData["FftData"]),1))
                Data["Timestamp"] = np.zeros((len(FFTData["FftData"]),1))
                Data["UnixTime"] = np.zeros((len(FFTData["FftData"]),1))
                Data["PacketGenTime"] = np.zeros((len(FFTData["FftData"]),1))
                Data["Sequences"] = np.zeros((len(FFTData["FftData"]),1))
                Data["Channels"] = np.zeros((len(FFTData["FftData"]),1))
                Data["Size"] = np.zeros((len(FFTData["FftData"]),1))
                Data["SamplingRate"] = np.zeros((len(FFTData["FftData"]),1))
                Data["Data"] = [None]*len(FFTData["FftData"])
                
                for PacketID in range(len(FFTData["FftData"])):
                    Data["SystemTick"][PacketID] = FFTData["FftData"][PacketID]["Header"]["systemTick"]
                    Data["Timestamp"][PacketID] = FFTData["FftData"][PacketID]["Header"]["timestamp"]["seconds"]
                    Data["UnixTime"][PacketID] = FFTData["FftData"][PacketID]["PacketRxUnixTime"]
                    Data["PacketGenTime"][PacketID] = FFTData["FftData"][PacketID]["PacketGenTime"]
                    Data["Sequences"][PacketID] = FFTData["FftData"][PacketID]["Header"]["dataTypeSequence"]
                    Data["Channels"][PacketID] = FFTData["FftData"][PacketID]["Channel"]
                    Data["Size"][PacketID] = FftSizes[FFTData["FftData"][PacketID]["FftSize"]] / 2
                    Data["SamplingRate"][PacketID] = TdSampleRates[FFTData["FftData"][PacketID]["SampleRate"]]
                    Data["Data"][PacketID] = np.array(FFTData["FftData"][PacketID]["FftOutput"]).T
                    
                # Correct for Missing Timestamps
                Data["SystemTick"] = unwrap(Data["SystemTick"], cap=65536).flatten() * 0.0001
                Data["Timestamp"] = Data["Timestamp"].flatten()
                Data["UnixTime"] = Data["UnixTime"].flatten() / 1000
                Data["PacketGenTime"] = Data["PacketGenTime"].flatten() / 1000
                Data["Sequences"] = unwrap(Data["Sequences"], cap=256).flatten()
                Data["Channels"] = Data["Channels"].flatten()
                Data["Size"] = Data["Size"].flatten()
                Data["SamplingRate"] = Data["SamplingRate"].flatten()
                Data["Time"] = Data["UnixTime"] - Data["UnixTime"][0]
        FFTDatas.append(Data)
        
    return FFTDatas

def fixMissingPacket_FFT(Data, StreamRate, PacketSize):
    FixedData = copy.deepcopy(Data)
    FixedData["Sequences"] = FixedData["Sequences"].astype(np.float64)
    
    # Adjust SystemTicks based on Timestamp
    for PacketID in range(1, len(FixedData["Timestamp"])):
        if FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID - 1] > 6:
            SecondsDifference = FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID-1]
            skippedLaps = np.round((SecondsDifference - (FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1])) / 6.5536)
            FixedData["SystemTick"][PacketID:] += skippedLaps * 6.5536
            SequenceJump = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1]) / StreamRate)
            FixedData["Sequences"][PacketID:] += (SequenceJump - (FixedData["Sequences"][PacketID] - FixedData["Sequences"][PacketID-1]))

    # Remove Bad Packets    
    SelectedPackets = np.concatenate(([0], np.where(np.diff(FixedData["SystemTick"]) > 0)[0].astype(int) + 1))
    DroppedPackets = np.where(np.diff(FixedData["SystemTick"]) <= 0)[0].astype(int) + 1
    adjustedIndex = 0
    for PacketID in DroppedPackets:
        FixedData["Data"].pop(PacketID - adjustedIndex)
        adjustedIndex += 1
        
    for field in FixedData.keys():
        if field == "Channels":
            FixedData[field] = FixedData[field][SelectedPackets]
        elif field != "Data" and field != "Configuration":
            FixedData[field] = FixedData[field][SelectedPackets]
    
    # Calculating the Drift Coefficient
    FixedData["SystemTick"] -= FixedData["SystemTick"][0]
    UnixTimer = FixedData["UnixTime"] - FixedData["UnixTime"][0]
    coefficients = np.polynomial.polynomial.polyfit(UnixTimer, FixedData["SystemTick"] - UnixTimer, 1)
    FixedData["DriftSlope"] = 1 / (1 + coefficients[1]);
    
    # Use the SystemTicks as adjustment for missing packets
    PacketID = 1
    while PacketID < len(FixedData["SystemTick"]):
        skipSequence = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID - 1]) / StreamRate).astype(int)
        if skipSequence == 0: 
            skipSequence = 1
        if skipSequence > 1:
            FixedData["SystemTick"] = np.insert(FixedData["SystemTick"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["UnixTime"] = np.insert(FixedData["UnixTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["PacketGenTime"] = np.insert(FixedData["PacketGenTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Timestamp"] = np.insert(FixedData["Timestamp"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Size"] = np.insert(FixedData["Size"], PacketID, PacketSize * np.ones(skipSequence - 1))
            FixedData["Channels"] = np.insert(FixedData["Channels"], PacketID, FixedData["Channels"][0] * np.ones((skipSequence - 1)), axis=0)
            FixedData["SamplingRate"] = np.insert(FixedData["SamplingRate"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Sequences"] = np.insert(FixedData["Sequences"], PacketID, -1 * np.ones(skipSequence - 1))
            for i in range(skipSequence-1):
                FixedData["Data"].insert(int(PacketID), np.zeros((int(PacketSize),1)))
        PacketID += skipSequence
    
    # Extract the FFT data to ndarray
    FixedData["Spectrogram"] = np.zeros((len(FixedData["Size"]), int(PacketSize)))
    FixedData["Missing"] = np.zeros((len(FixedData["Size"]), 1))
    for n in range(len(FixedData["Data"])):
        if FixedData["SystemTick"][n] == -1:
            FixedData["Missing"][n,:] = 1
        FixedData["Spectrogram"][n,:] = np.array(FixedData["Data"][n]).flatten()
    FixedData["Time"] = np.array(range(len(FixedData["Data"]))) * StreamRate
    
    return FixedData

def getPowerData(DataFolder):
    JSON = decodeJSON(DataFolder + "RawDataPower.json", fileType="RawDataPower")
    PowerDatas = list()
    for PowerData in JSON:
        Data = dict()
        if "PowerDomainData" in PowerData.keys():
            if PowerData["PowerDomainData"] != []:
                Data["SystemTick"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["Timestamp"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["UnixTime"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["PacketGenTime"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["Sequences"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["FFTSize"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["SamplingRate"] = np.zeros((len(PowerData["PowerDomainData"]),1))
                Data["DataMasks"] = np.zeros((len(PowerData["PowerDomainData"]),2))
                Data["Data"] = [None]*len(PowerData["PowerDomainData"])
                
                for PacketID in range(len(PowerData["PowerDomainData"])):
                    Data["SystemTick"][PacketID] = PowerData["PowerDomainData"][PacketID]["Header"]["systemTick"]
                    Data["Timestamp"][PacketID] = PowerData["PowerDomainData"][PacketID]["Header"]["timestamp"]["seconds"]
                    Data["UnixTime"][PacketID] = PowerData["PowerDomainData"][PacketID]["PacketRxUnixTime"]
                    Data["PacketGenTime"][PacketID] = PowerData["PowerDomainData"][PacketID]["PacketGenTime"]
                    Data["Sequences"][PacketID] = PowerData["PowerDomainData"][PacketID]["Header"]["dataTypeSequence"]
                    Data["DataMasks"][PacketID,:] = [PowerData["PowerDomainData"][PacketID]["ExternalValuesMask"],
                                                   PowerData["PowerDomainData"][PacketID]["ValidDataMask"]]
                    Data["FFTSize"][PacketID] = FftSizes[PowerData["PowerDomainData"][PacketID]["FftSize"]] / 2
                    Data["SamplingRate"][PacketID] = TdSampleRates[PowerData["PowerDomainData"][PacketID]["SampleRate"]]
                    Data["Data"][PacketID] = np.array(PowerData["PowerDomainData"][PacketID]["Bands"]).T
                    
                # Correct for Missing Timestamps
                Data["SystemTick"] = unwrap(Data["SystemTick"], cap=65536).flatten() * 0.0001
                Data["Timestamp"] = Data["Timestamp"].flatten()
                Data["UnixTime"] = Data["UnixTime"].flatten() / 1000
                Data["PacketGenTime"] = Data["PacketGenTime"].flatten() / 1000
                Data["Sequences"] = unwrap(Data["Sequences"], cap=256).flatten()
                Data["FFTSize"] = Data["FFTSize"].flatten()
                Data["SamplingRate"] = Data["SamplingRate"].flatten()
                
        PowerDatas.append(Data)
        
    return PowerDatas

def fixMissingPacket_Power(Data, StreamRate):
    FixedData = copy.deepcopy(Data)
    FixedData["Sequences"] = FixedData["Sequences"].astype(np.float64)
    
    # Adjust SystemTicks based on Timestamp
    for PacketID in range(1, len(FixedData["Timestamp"])):
        if FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID - 1] > 6:
            SecondsDifference = FixedData["Timestamp"][PacketID] - FixedData["Timestamp"][PacketID-1]
            skippedLaps = np.round((SecondsDifference - (FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1])) / 6.5536)
            FixedData["SystemTick"][PacketID:] += skippedLaps * 6.5536
            SequenceJump = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID-1]) / StreamRate)
            FixedData["Sequences"][PacketID:] += (SequenceJump - (FixedData["Sequences"][PacketID] - FixedData["Sequences"][PacketID-1]))

    # Remove Bad Packets    
    SelectedPackets = np.concatenate(([0], np.where(np.diff(FixedData["SystemTick"]) > 0)[0].astype(int) + 1))
    DroppedPackets = np.where(np.diff(FixedData["SystemTick"]) <= 0)[0].astype(int) + 1
    adjustedIndex = 0
    for PacketID in DroppedPackets:
        FixedData["Data"].pop(PacketID - adjustedIndex)
        adjustedIndex += 1
        
    for field in FixedData.keys():
        if field != "Data" and field != "Configuration":
            FixedData[field] = FixedData[field][SelectedPackets]
    
    # Calculating the Drift Coefficient
    FixedData["SystemTick"] -= FixedData["SystemTick"][0]
    UnixTimer = FixedData["UnixTime"] - FixedData["UnixTime"][0]
    coefficients = np.polynomial.polynomial.polyfit(UnixTimer, FixedData["SystemTick"] - UnixTimer, 1)
    FixedData["DriftSlope"] = 1 / (1 + coefficients[1]);
    
    # Use the SystemTicks as adjustment for missing packets
    PacketID = 1
    while PacketID < len(FixedData["SystemTick"]):
        skipSequence = np.round((FixedData["SystemTick"][PacketID] - FixedData["SystemTick"][PacketID - 1]) / StreamRate).astype(int)
        if skipSequence == 0: 
            skipSequence = 1
        if skipSequence > 1:
            FixedData["SystemTick"] = np.insert(FixedData["SystemTick"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["UnixTime"] = np.insert(FixedData["UnixTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["PacketGenTime"] = np.insert(FixedData["PacketGenTime"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Timestamp"] = np.insert(FixedData["Timestamp"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["FFTSize"] = np.insert(FixedData["FFTSize"], PacketID, FixedData["FFTSize"][0] * np.ones(skipSequence - 1))
            FixedData["DataMasks"] = np.insert(FixedData["DataMasks"], PacketID, -1 * np.ones((skipSequence - 1, 2)), axis=0)
            FixedData["SamplingRate"] = np.insert(FixedData["SamplingRate"], PacketID, -1 * np.ones(skipSequence - 1))
            FixedData["Sequences"] = np.insert(FixedData["Sequences"], PacketID, -1 * np.ones(skipSequence - 1))
            for i in range(skipSequence-1):
                FixedData["Data"].insert(int(PacketID), -1 * np.zeros((8,1)))
        PacketID += skipSequence
    
    # Extract the FFT data to ndarray
    FixedData["Power"] = np.zeros((len(FixedData["FFTSize"]), 8))
    FixedData["Missing"] = np.zeros((len(FixedData["FFTSize"]), 1))
    for n in range(len(FixedData["Data"])):
        if FixedData["SystemTick"][n] == -1:
            FixedData["Missing"][n,:] = 1
        FixedData["Power"][n,:] = np.array(FixedData["Data"][n]).flatten()
    FixedData["Time"] = np.array(range(len(FixedData["Data"]))) * StreamRate
    
    return FixedData

def getTimeSyncData(DataFolder):
    JSON = decodeJSON(DataFolder + "TimeSync.json", fileType="TimeSync")
    TimeSyncDatas = list()
    for TimeSyncData in JSON:
        Data = dict()
        if "TimeSyncData" in TimeSyncData.keys():
            if TimeSyncData["TimeSyncData"] != []:
                Data["SystemTick"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                Data["Timestamp"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                Data["UnixTime"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                Data["PacketGenTime"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                Data["Sequences"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                Data["Latency"] = np.zeros((len(TimeSyncData["TimeSyncData"]),1))
                for PacketID in range(len(TimeSyncData["TimeSyncData"])):
                    Data["SystemTick"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["Header"]["systemTick"]
                    Data["Timestamp"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["Header"]["timestamp"]["seconds"]
                    Data["UnixTime"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["PacketRxUnixTime"]
                    Data["PacketGenTime"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["PacketGenTime"]
                    Data["Sequences"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["Header"]["dataTypeSequence"]
                    Data["Latency"][PacketID] = TimeSyncData["TimeSyncData"][PacketID]["LatencyMilliseconds"]
                    
                # Correct for Missing Timestamps
                Data["SystemTick"] = unwrap(Data["SystemTick"], cap=65536).flatten() * 0.0001
                Data["Timestamp"] = Data["Timestamp"].flatten()
                Data["Sequences"] = unwrap(Data["Sequences"], cap=256).flatten()
                Data["UnixTime"] = Data["UnixTime"].flatten() / 1000
                Data["Latency"] = Data["Latency"].flatten()
            TimeSyncDatas.append(Data)
    return TimeSyncDatas

def getAdaptiveStimData(DataFolder):
    JSON = decodeJSON(DataFolder + "AdaptiveLog.json", fileType="AdaptiveLog")
    JSON = [package for package in JSON if "AdaptiveUpdate" in package.keys()]
    
    if len(JSON) == 0:
        return {}
    
    Data = dict()
    Data["SystemTick"] = np.zeros((len(JSON),1))
    Data["Timestamp"] = np.zeros((len(JSON),1))
    Data["UnixTime"] = np.zeros((len(JSON),1))
    Data["PacketGenTime"] = np.zeros((len(JSON),1))
    Data["Sequences"] = np.zeros((len(JSON),1))
    Data["CurrentAdaptiveState"] = np.zeros((len(JSON),1))
    Data["StimRateInHz"] = np.zeros((len(JSON),1))
    Data["Amplitude"] = np.zeros((len(JSON),4))
    
    Data["Ld0"] = [package["AdaptiveUpdate"]["Ld0Status"] for package in JSON]
    Data["Ld1"] = [package["AdaptiveUpdate"]["Ld1Status"] for package in JSON]
    
    for PacketID in range(len(JSON)):
        Data["SystemTick"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["Header"]["systemTick"]
        Data["Timestamp"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["Header"]["timestamp"]["seconds"]
        Data["UnixTime"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["PacketRxUnixTime"]
        Data["PacketGenTime"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["PacketGenTime"]
        Data["Sequences"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["Header"]["dataTypeSequence"]
        Data["CurrentAdaptiveState"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["CurrentAdaptiveState"]
        Data["StimRateInHz"][PacketID] = JSON[PacketID]["AdaptiveUpdate"]["StimRateInHz"]
        Data["Amplitude"][PacketID,:] = JSON[PacketID]["AdaptiveUpdate"]["CurrentProgramAmplitudesInMilliamps"]
        
        
    # Correct for Missing Timestamps
    Data["SystemTick"] = unwrap(Data["SystemTick"], cap=65536).flatten() * 0.0001
    Data["Timestamp"] = Data["Timestamp"].flatten()
    Data["UnixTime"] = Data["UnixTime"].flatten() / 1000
    Data["PacketGenTime"] = Data["PacketGenTime"].flatten() / 1000
    Data["Sequences"] = unwrap(Data["Sequences"], cap=256).flatten()
    Data["CurrentAdaptiveState"] = Data["CurrentAdaptiveState"].flatten()
    Data["StimRateInHz"] = Data["StimRateInHz"].flatten()
    
    return Data

def getStimulationLogs(DataFolder):
    JSON = decodeJSON(DataFolder + "StimLog.json", fileType="StimLog")
    StimulationLogs = list()
    for StimData in JSON:
        Data = dict()
        
        # Telemetry Info (When is the following configuration transmitted)
        if "RecordInfo" in StimData.keys():
            Data["Time"] = StimData["RecordInfo"]["HostUnixTime"] / 1000.0
            # TODO: Add UTC Offset
            Data["SessionDateTime"] = datetime.fromtimestamp(float(StimData["RecordInfo"]["SessionId"]) / 1000.0) 
        
        if "therapyStatusData" in StimData.keys():
            if "activeGroup" in StimData["therapyStatusData"].keys():
                Data["ActiveGroup"] = StimData["therapyStatusData"]["activeGroup"]
                
            if "therapyStatus" in StimData["therapyStatusData"].keys():
                Data["TherapyStatus"] = StimData["therapyStatusData"]["therapyStatus"]
        
        Data["TherapyGroups"] = list()
        for n in range(4):
            TherapyConfig = dict()
            if "TherapyConfigGroup" + str(n) in StimData.keys():
                if "RateInHz" in StimData["TherapyConfigGroup" + str(n)].keys():
                    TherapyConfig["Frequency"] = StimData["TherapyConfigGroup" + str(n)]["RateInHz"]
                    
                TherapyConfig["Programs"] = list()
                for i in range(4):
                    ProgramSetting = dict()
                    if "program"+str(i) in StimData["TherapyConfigGroup" + str(n)].keys():
                        if "AmplitudeInMilliamps" in StimData["TherapyConfigGroup" + str(n)]["program"+str(i)].keys():
                            ProgramSetting["Amplitude"] = StimData["TherapyConfigGroup" + str(n)]["program"+str(i)]["AmplitudeInMilliamps"]
                        if "PulseWidthInMicroseconds" in StimData["TherapyConfigGroup" + str(n)]["program"+str(i)].keys():
                            ProgramSetting["PulseWidth"] = StimData["TherapyConfigGroup" + str(n)]["program"+str(i)]["PulseWidthInMicroseconds"]
                    TherapyConfig["Programs"].append(ProgramSetting)
            Data["TherapyGroups"].append(TherapyConfig)
        StimulationLogs.append(Data)
    return StimulationLogs

def getEventLogs(DataFolder):
    JSON = decodeJSON(DataFolder + "EventLog.json", fileType="EventLog")
    EventLogs = list()
    for EventDescription in JSON:
        Data = dict()
        
        # Telemetry Info (When is the following configuration transmitted)
        if "RecordInfo" in EventDescription.keys():
            Data["Time"] = EventDescription["RecordInfo"]["HostUnixTime"] / 1000.0
            # TODO: Add UTC Offset
            Data["SessionDateTime"] = datetime.fromtimestamp(float(EventDescription["RecordInfo"]["SessionId"]) / 1000.0) 
        
        if "Event" in EventDescription.keys():
            Data["Event"] = EventDescription["Event"]
        
        EventLogs.append(Data)
    return EventLogs

def getErrorLogs(DataFolder):
    JSON = decodeJSON(DataFolder + "ErrorLog.json", fileType="ErrorLog")
    ErrorLogs = list()
    for ErrorDescription in JSON:
        Data = dict()
        
        # Telemetry Info (When is the following configuration transmitted)
        if "RecordInfo" in ErrorDescription.keys():
            Data["Time"] = ErrorDescription["RecordInfo"]["HostUnixTime"] / 1000.0
            # TODO: Add UTC Offset
            Data["SessionDateTime"] = datetime.fromtimestamp(float(ErrorDescription["RecordInfo"]["SessionId"]) / 1000.0) 
        
        if "CommandType" in ErrorDescription.keys():
            Data["Command"] = ErrorDescription["CommandType"]
            
        if "Error" in ErrorDescription.keys():
            Data["Error"] = ErrorDescription["Error"]
        
        ErrorLogs.append(Data)
    return ErrorLogs

def getAdaptiveSensingLog(DataFolder, timeshift):
    logFolderName = ""
    
    for folder in os.listdir(DataFolder):
        if folder.startswith("LogDataFrom"):
            logFolderName = folder
    
    if logFolderName == "":
        return []
    
    LogFiles = os.listdir(DataFolder + logFolderName)
    for file in LogFiles:
        if file.endswith("AppLog.txt"):
            with open(DataFolder + logFolderName + "/" + file, "r") as fp:
                LogContent = fp.readlines()
                
            lineNum = 0
            AllLogs = []
            LogEntry = {}
            while lineNum < len(LogContent):
                try: 
                    if "LogEntry.Header" in LogContent[lineNum]:
                        AllLogs.append(LogEntry)
                        LogEntry = {}
                    
                    if "LogHeader.EntryStatus" in LogContent[lineNum]:
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Status"] = int(components[i+1], 16) 
                                break
                    
                    if "LogHeader.EntryTimestamp" in LogContent[lineNum]:
                        lineNum += 1
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Timestamp"] = int(components[i+1].strip(","), 16) + 946702800 - timeshift
                                break
                    
                    if "LogEntry.Payload" in LogContent[lineNum]:
                        LogEntry["Payload"] = {}
                    
                    if "EventId" in LogContent[lineNum] and "CommonLogPayload" in LogContent[lineNum]:
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Payload"]["ID"] = EventIDs[int(components[i+1], 16)]
                                break
                    
                    if "EntryPayload" in LogContent[lineNum] and "CommonLogPayload" in LogContent[lineNum]:
                        if LogEntry["Payload"]["ID"] == "AdaptiveTherapyStateChange":
                            LogEntry["Payload"]["AdaptiveTherapy"] = {}
                            lineNum += 1
                            while "AdaptiveTherapyModificationEntry" in LogContent[lineNum]:
                                if ".NewState" in LogContent[lineNum]:
                                    components = LogContent[lineNum].split(" ")
                                    for i in range(len(components)):
                                        if components[i] == "=":
                                            LogEntry["Payload"]["AdaptiveTherapy"]["NewState"] = int(components[i+1], 16)
                                            break
                                elif ".OldState" in LogContent[lineNum]:
                                    components = LogContent[lineNum].split(" ")
                                    for i in range(len(components)):
                                        if components[i] == "=":
                                            LogEntry["Payload"]["AdaptiveTherapy"]["OldState"] = int(components[i+1], 16)
                                            break
                                        
                                lineNum += 1
                except Exception as e:
                    print(e)
                    
                lineNum += 1
            
            filteredLogs = []
            for log in AllLogs:
                if "Payload" in log.keys():
                    if "ID" in log["Payload"].keys():
                        filteredLogs.append(log)
                        
            return filteredLogs[-1:0:-1]

def getSystemEventLogs(DataFolder, timeshift=0):
    logFolderName = ""
    
    for folder in os.listdir(DataFolder):
        if folder.startswith("LogDataFrom"):
            logFolderName = folder
            
    if logFolderName == "":
        return []
    
    LogFiles = os.listdir(DataFolder + logFolderName)
    for file in LogFiles:
        if file.endswith("EventLog.txt"):
            with open(DataFolder + logFolderName + "/" + file, "r") as fp:
                LogContent = fp.readlines()
                
            lineNum = 0
            AllLogs = []
            LogEntry = {}
            while lineNum < len(LogContent):
                try: 
                    if "LogEntry.Header" in LogContent[lineNum]:
                        AllLogs.append(LogEntry)
                        LogEntry = {}
                    
                    if "LogHeader.EntryStatus" in LogContent[lineNum]:
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Status"] = int(components[i+1], 16) 
                                break
                    
                    if "LogHeader.EntryTimestamp" in LogContent[lineNum]:
                        lineNum += 1
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Timestamp"] = int(components[i+1].strip(","), 16) + 946702800 - timeshift
                                break
                    
                    if "LogEntry.Payload" in LogContent[lineNum]:
                        LogEntry["Payload"] = {}
                    
                    if "EventId" in LogContent[lineNum] and "CommonLogPayload" in LogContent[lineNum]:
                        components = LogContent[lineNum].split(" ")
                        for i in range(len(components)):
                            if components[i] == "=":
                                LogEntry["Payload"]["ID"] = EventIDs[int(components[i+1], 16)]
                                break
                    
                    if "EntryPayload" in LogContent[lineNum] and "CommonLogPayload" in LogContent[lineNum]:
                    
                        if LogEntry["Payload"]["ID"] == "TherapyStatus":
                            LogEntry["Payload"]["TherapyEvent"] = {}
                            lineNum += 1
                            while "TherapyStatusEventLogEntry" in LogContent[lineNum]:
                                if ".TherapyStatusType" in LogContent[lineNum]:
                                    components = LogContent[lineNum].split(" ")
                                    for i in range(len(components)):
                                        if components[i] == "=":
                                            LogEntry["Payload"]["TherapyEvent"]["Type"] = int(components[i+1], 16)
                                            break
                                elif ".TherapyStatus" in LogContent[lineNum]:
                                    components = LogContent[lineNum].split(" ")
                                    for i in range(len(components)):
                                        if components[i] == "=":
                                            LogEntry["Payload"]["TherapyEvent"]["Status"] = int(components[i+1], 16)
                                            break
                                        
                                lineNum += 1
                except Exception as e:
                    print(e)
                    
                lineNum += 1
            return AllLogs[1:]

def findLatestConfiguration(Configurations, UnixTimeOfFirstPacket, Type):
    TargetConfiguration = dict()
    for config in Configurations:
        if Type == "AdaptiveStim":
            if config["Time"] < UnixTimeOfFirstPacket:
                combinedConfig = dict()
                if "Detector" in config.keys() or "Adaptive" in config.keys() or "SensingConfiguration" in config.keys():
                    TargetConfiguration = dict()
                    if "Detector" in config.keys():
                        TargetConfiguration["Detector"] = config["Detector"]
                    if "Adaptive" in config.keys():
                        TargetConfiguration["Adaptive"] = config["Adaptive"]
        
        elif "SensingConfiguration" in config.keys() and config["Time"] < UnixTimeOfFirstPacket:
            if Type == "Power":
                if "FFT" in config["SensingConfiguration"].keys():
                    TargetConfiguration = config["SensingConfiguration"]["FFT"]
                if "Power" in config["SensingConfiguration"].keys():
                    TargetConfiguration["PowerBands"] = config["SensingConfiguration"]["Power"].tolist()
                if "PowerEnable" in config["SensingConfiguration"].keys():
                    TargetConfiguration["Enabled"] = config["SensingConfiguration"]["PowerEnable"]
                if "Lfp" in config["SensingConfiguration"].keys():
                    TargetConfiguration["LfpConfig"] = config["SensingConfiguration"]["Lfp"]
            else:
                if Type in config["SensingConfiguration"].keys():
                    TargetConfiguration = config["SensingConfiguration"][Type]
                    
            if Type == "Lfp":
                if "Misc" in config["SensingConfiguration"].keys():
                    for channelConfig in TargetConfiguration:
                        channelConfig["StreamRate"] = config["SensingConfiguration"]["Misc"]["StreamRate"]
            
        # FFT Configuration has 1 extra information: Streaming Channel, which is only available in SenseStates
        if Type == "FFT":
            if "SenseStates" in config.keys() and config["Time"] < UnixTimeOfFirstPacket:
                if "FFTChannel" in config["SenseStates"].keys():
                    TargetConfiguration["FFTChannel"] = config["SenseStates"]["FFTChannel"]
        
    return TargetConfiguration

# The basic idea behind this function is to handle None type object and Datetime objects that MATLAB can't read
def saveSummitDataToMATLAB(filename, Data):
    
    # Create a deep copy of the object so we don't mess with the original data
    matData = copy.deepcopy(Data)
    
    for key in Data.keys():
        if key == "Config" or key == "StimLogs" or key == "ErrorLogs" or key == "EventLogs":
            for Config in matData[key]:
                if "SessionDateTime" in Config:
                    Config["SessionDateTime"] = Config["SessionDateTime"].isoformat()
                    
        if key == "Lfp":
            for trial in range(len(matData[key])):
                
                # EvokedMarker may contain None
                for i in range(len(matData[key][trial]["EvokedMarker"])):
                    if matData[key][trial]["EvokedMarker"][i] == []:
                        matData[key][trial]["EvokedMarker"][i] = np.nan
                    if matData[key][trial]["EvokedMarker"][i] == [None]:
                        matData[key][trial]["EvokedMarker"][i] = np.nan
                
                # DebugInfo may contain None
                for i in range(len(matData[key][trial]["DebugInfo"])):
                    if matData[key][trial]["DebugInfo"][i] == []:
                        matData[key][trial]["DebugInfo"][i] = np.nan
                    if matData[key][trial]["DebugInfo"][i] == [None]:
                        matData[key][trial]["DebugInfo"][i] = np.nan

        if key == "FFT" or key == "Power" or key == "Lfp":
            for trial in range(len(matData[key])):
                # Drop Data to save space
                del(matData[key][trial]["Data"])
        # Skip FFTs. Nothing in FFTs can be None
    
    sio.savemat(filename, matData, long_field_names=True)
    
def LoadData(DataFolder, DataType=["Lfp", "FFT", "Power", "Accelerometer", "AdaptiveStim", "DetectionLd", "LoopRecording", "TimeSync"]):
    for file in os.listdir(DataFolder):
        if file.startswith("Device") and os.path.isdir(DataFolder + "/" + file):
            DataFolder = DataFolder + "/" + file + "/"

    Data = dict()
    Data["Config"] = getDeviceSettings(DataFolder)
            
    StreamTime = dict()
    for key in DataType:
        StreamTime[key] = list()
        
    for Configuration in Data["Config"]:
        if "StreamState" in Configuration.keys():
            for key in DataType:
                if key in Configuration["StreamState"]:
                    StreamTime[key].append(Configuration["Time"])
                    
    for key in DataType:
        if len(StreamTime[key]) == 0:
            continue
    
        # Time Domain Segments
        if key == "Lfp":
            rawTD = getTimeDomainData(DataFolder)
            
            if rawTD == [{}]: 
                continue
            
            Data[key] = list()
            for n in range(len(StreamTime[key])):
                Data[key].append(copy.deepcopy(rawTD[0]))
                if n < len(StreamTime[key]) - 1:
                    SegmentSelection = np.bitwise_and(Data[key][n]["UnixTime"] > StreamTime[key][n], Data[key][n]["UnixTime"] < StreamTime[key][n+1])
                else:
                    SegmentSelection = Data[key][n]["UnixTime"] > StreamTime[key][n]
                
                if np.any(SegmentSelection):
                    for field in Data[key][n].keys():
                        if type(Data[key][n][field]) == list:
                            indexes = np.where(SegmentSelection)[0].astype(int)
                            Data[key][n][field] = Data[key][n][field][slice(indexes[0], indexes[-1]+1, 1)]
                        else:
                            Data[key][n][field] = Data[key][n][field][SegmentSelection]
                    
                    Data[key][n]["Configuration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], key)
                    Data[key][n] = fixMissingPacket_TD(Data[key][n], 
                                                       Data[key][n]["Configuration"][0]["StreamRate"] / 1000, 
                                                       int(Data[key][n]["Configuration"][0]["StreamRate"] / 1000 * Data[key][n]["SamplingRate"][0]))
    
        # FFT Segments
        if key == "FFT":
            rawFFT = getFFTData(DataFolder)
            
            if rawFFT == [{}]: 
                continue
            
            Data[key] = list()
            for n in range(len(StreamTime[key])):
                Data[key].append(copy.deepcopy(rawFFT[0]))
                if n < len(StreamTime[key]) - 1:
                    SegmentSelection = np.bitwise_and(Data[key][n]["UnixTime"] > StreamTime[key][n], Data[key][n]["UnixTime"] < StreamTime[key][n+1])
                else:
                    SegmentSelection = Data[key][n]["UnixTime"] > StreamTime[key][n]
                
                if np.any(SegmentSelection):
                    for field in Data[key][n].keys():
                        if type(Data[key][n][field]) == list:
                            indexes = np.where(SegmentSelection)[0].astype(int)
                            Data[key][n][field] = Data[key][n][field][slice(indexes[0], indexes[-1]+1, 1)]
                        else:
                            Data[key][n][field] = Data[key][n][field][SegmentSelection]
                    
                    Data[key][n]["Configuration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], key)
                    Data[key][n] = fixMissingPacket_FFT(Data[key][n], 
                                                          Data[key][n]["Configuration"]["Interval"] / 1000, 
                                                          Data[key][n]["Configuration"]["StreamBinSize"])
                    
                    SamplingRate = Data[key][n]["SamplingRate"][0]
                    FrequencyResolution = SamplingRate / Data[key][n]["Configuration"]["NFFT"] 
                    Data[key][n]["Frequency"] = np.array(range(int(Data[key][n]["Configuration"]["StreamOffset"]),int(Data[key][n]["Configuration"]["StreamBinSize"]))) * FrequencyResolution

        # Power Segments
        if key == "Power":
            rawPower = getPowerData(DataFolder)
            
            if rawPower == [{}]: 
                continue
            
            Data[key] = list()
            for n in range(len(StreamTime[key])):
                Data[key].append(copy.deepcopy(rawPower[0]))
                if n < len(StreamTime[key]) - 1:
                    SegmentSelection = np.bitwise_and(Data[key][n]["UnixTime"] > StreamTime[key][n], Data[key][n]["UnixTime"] < StreamTime[key][n+1])
                else:
                    SegmentSelection = Data[key][n]["UnixTime"] > StreamTime[key][n]
                
                if np.any(SegmentSelection):
                    for field in Data[key][n].keys():
                        if type(Data[key][n][field]) == list:
                            indexes = np.where(SegmentSelection)[0].astype(int)
                            Data[key][n][field] = Data[key][n][field][slice(indexes[0], indexes[-1]+1, 1)]
                        else:
                            Data[key][n][field] = Data[key][n][field][SegmentSelection]
                    
                    Data[key][n]["Configuration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], key)
                    Data[key][n] = fixMissingPacket_Power(Data[key][n], 
                                                          Data[key][n]["Configuration"]["Interval"] / 1000)
                    Data[key][n]["ChannelConfiguration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], "Lfp")
                    
        # Time Sync Segments
        if key == "TimeSync":
            rawTimeSync = getTimeSyncData(DataFolder)
            Data["TimeSync"] = list()
            for n in range(len(StreamTime["TimeSync"])):
                Data["TimeSync"].append(copy.deepcopy(rawTimeSync[0]))
                if n < len(StreamTime["TimeSync"]) - 1:
                    SegmentSelection = np.bitwise_and(Data["TimeSync"][n]["UnixTime"] > StreamTime["TimeSync"][n], Data["TimeSync"][n]["UnixTime"] < StreamTime["TimeSync"][n+1])
                else:
                    SegmentSelection = Data["TimeSync"][n]["UnixTime"] > StreamTime["TimeSync"][n]
                    
                Data["TimeSync"][n]["SystemTick"] = Data["TimeSync"][n]["SystemTick"][SegmentSelection]
                Data["TimeSync"][n]["Timestamp"] = Data["TimeSync"][n]["Timestamp"][SegmentSelection]
                Data["TimeSync"][n]["UnixTime"] = Data["TimeSync"][n]["UnixTime"][SegmentSelection]
                Data["TimeSync"][n]["Sequences"] = Data["TimeSync"][n]["Sequences"][SegmentSelection]
                Data["TimeSync"][n]["Latency"] = Data["TimeSync"][n]["Latency"][SegmentSelection]
                
        if key == "AdaptiveStim":
            rawAdaptiveStimInfo = getAdaptiveStimData(DataFolder)
            if rawAdaptiveStimInfo == {}: 
                continue
            
            Data[key] = list()
            for n in range(len(StreamTime["AdaptiveStim"])):
                Data["AdaptiveStim"].append(copy.deepcopy(rawAdaptiveStimInfo))
                if n < len(StreamTime["AdaptiveStim"]) - 1:
                    SegmentSelection = np.bitwise_and(Data["AdaptiveStim"][n]["UnixTime"] > StreamTime["AdaptiveStim"][n], Data["AdaptiveStim"][n]["UnixTime"] < StreamTime["AdaptiveStim"][n+1])
                else:
                    SegmentSelection = Data["AdaptiveStim"][n]["UnixTime"] > StreamTime["AdaptiveStim"][n]
                
                if np.any(SegmentSelection):
                    for field in Data[key][n].keys():
                        if type(Data[key][n][field]) == list:
                            indexes = np.where(SegmentSelection)[0].astype(int)
                            Data[key][n][field] = Data[key][n][field][slice(indexes[0], indexes[-1]+1, 1)]
                        else:
                            Data[key][n][field] = Data[key][n][field][SegmentSelection]

                    #Data[key][n]["Configuration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], key)
                    Data[key][n]["Configuration"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], key)
                    Data[key][n]["Configuration"]["Power"] = findLatestConfiguration(Data["Config"], Data[key][n]["UnixTime"][0], "Power")
    
    try:
        HostUnixTime = Data["Config"][0]["Time"]
        DeviceTime = Data["Config"][0]["GeneralData"]["deviceTime"]["seconds"] + 946702800
        Timeshift = DeviceTime - HostUnixTime

        Data["StimLogs"] = getStimulationLogs(DataFolder)
        Data["EventLogs"] = getEventLogs(DataFolder)
        Data["ErrorLogs"] = getErrorLogs(DataFolder)
        Data["SystemLogs"] = getSystemEventLogs(DataFolder, timeshift=Timeshift)
        Data["AdaptiveLogs"] = getAdaptiveSensingLog(DataFolder, timeshift=Timeshift)
    except Exception as e:
        pass
    
    return Data

def PrintSessionInfo(Data):
    print("Session UTC Date: {0}".format(Data["Config"][0]["SessionDateTime"]))
    
    if "Lfp" in Data.keys():
        print(f"\nNumber of LFP Recordings: {len(Data['Lfp'])}")
        for n in range(len(Data["Lfp"])):
            print(f"\tDuration of LFP Recording #{n+1}: {Data['Lfp'][n]['Time'][-1] / 60:.2f} minutes")
            print(f"\tPercent Missing Data of LFP Recording #{n+1}: {100 * np.sum(Data['Lfp'][n]['Missing']) / len(Data['Lfp'][n]['Missing']):.2f}%")
    if "FFT" in Data.keys():
        print(f"\nNumber of FFT Recordings: {len(Data['FFT'])}")
    if "Power" in Data.keys():
        print(f"\nNumber of Power Recordings: {len(Data['Power'])}")

def PrintConfigurationInfo(Data):
    FirstStreamTime = Data["Config"][0]["Time"]
    ConfigurationID = 0
    for Configuration in Data["Config"]:
        TimeOccured = Configuration["Time"] - FirstStreamTime
        if "Therapy" in Configuration.keys():
            print("Time {0} - Config {1}: Therapy".format(TimeOccured,ConfigurationID))
            #print(Configuration["Therapy"])
        if "Adaptive" in Configuration.keys():
            print("Time {0} - Config {1}: Adaptive".format(TimeOccured,ConfigurationID))
        if "BatteryStatus" in Configuration.keys():
            print("Time {0} - Config {1}: BatteryStatus".format(TimeOccured,ConfigurationID))
        if "StreamState" in Configuration.keys():
            print("Time {0} - Config {1}: StreamState".format(TimeOccured,ConfigurationID))
        if "SenseStates" in Configuration.keys():
            print("Time {0} - Config {1}: SenseStates".format(TimeOccured,ConfigurationID))
        if "SensingConfiguration" in Configuration.keys():
            print("Time {0} - Config {1}: SensingConfiguration".format(TimeOccured,ConfigurationID))
        #if "TelemetryModule" in Configuration.keys():
        #    print("Time {0} - Config {1}: Telemetry Reconnection".format(TimeOccured,ConfigurationID))
        if "LeadLocation" in Configuration.keys():
            Data["LeadLocation"] = Configuration["LeadLocation"]
        ConfigurationID += 1

def TherapyReconstruction(Data):
    CurrentTherapy = copy.deepcopy(Data["Config"][0]["Therapy"])
    
    TherapyConfiguration = list()
    for log in Data["StimLogs"]:
        if "ActiveGroup" in log.keys():
            for groupID in range(len(CurrentTherapy)):
                CurrentTherapy[groupID]["ActiveGroup"] = log["ActiveGroup"] == groupID
        if "TherapyStatus" in log.keys():
            for groupID in range(len(CurrentTherapy)):
                CurrentTherapy[groupID]["TherapyStatus"] = log["TherapyStatus"]
        if "TherapyGroups" in log.keys():
            for groupID in range(len(CurrentTherapy)):
                if CurrentTherapy[groupID]["Valid"]:
                    if "Frequency" in log["TherapyGroups"][groupID].keys():
                        CurrentTherapy[groupID]["Frequency"] = log["TherapyGroups"][groupID]["Frequency"]
                    if "Programs" in log["TherapyGroups"][groupID].keys():
                        for programID in range(len(log["TherapyGroups"][groupID]["Programs"])):
                            if CurrentTherapy[groupID]["Programs"][programID]["Valid"]:
                                if "Amplitude" in log["TherapyGroups"][groupID]["Programs"][programID].keys():
                                    CurrentTherapy[groupID]["Programs"][programID]["Amplitude"] = log["TherapyGroups"][groupID]["Programs"][programID]["Amplitude"]
                                if "PulseWidth" in log["TherapyGroups"][groupID]["Programs"][programID].keys():
                                    CurrentTherapy[groupID]["Programs"][programID]["PulseWidth"] = log["TherapyGroups"][groupID]["Programs"][programID]["PulseWidth"]
        
        CurrentTherapy[3]["Adaptive"] = findLatestConfiguration(Data["Config"], log["Time"], "AdaptiveStim")
        CurrentTherapy[3]["Adaptive"]["Power"] = findLatestConfiguration(Data["Config"], log["Time"], "Power")
        TherapyConfiguration.append({"Time": log["Time"], "Therapy": copy.deepcopy(CurrentTherapy)})
    
    TherapyID = 1
    while TherapyID < len(TherapyConfiguration):
        if TherapyConfiguration[TherapyID]["Therapy"] == TherapyConfiguration[TherapyID-1]["Therapy"]:
            TherapyConfiguration.pop(TherapyID)
        else:
            TherapyID += 1
    
    return TherapyConfiguration

def TimeDomain2PowerChannel(data, time, interval, gain=250, nFFT=256, fs=250):

    FP_READ_UNITS_VALUE = 48644.8683623726;
    missingData = data == 0
    data -= np.mean(data[np.invert(missingData)])
    data[missingData] = 0
    data = data * (250*gain/255) * FP_READ_UNITS_VALUE / 1200
    
    if nFFT == 256:
        fftPts = 250
    elif nFFT == 1024:
        fftPts = 1000
    elif nFFT == 64:
        fftPts = 62
    else:
        fftPts = nFFT
    
    window = np.ceil(fs*interval/1000)
    overlap = fftPts - window
    hanningWindow = np.hanning(nFFT)
    nEpochs = int(np.ceil(len(data)/(window)))
    PowerChannel = np.zeros((int(nFFT/2),nEpochs))
    TimeRange = np.zeros((nEpochs))
    Frequency = np.arange(nFFT/2)*(fs/nFFT)
    
    for i in range(nEpochs):
        if i*window+nFFT < len(data):
            X = np.fft.fft(data[int(i*window):int(i*window+nFFT)] * hanningWindow, nFFT)
            SSB = X[:int(nFFT/2)]
            SSB[1:] *= 2
            YFFT = np.abs(SSB/(nFFT/2))
            fftPower = 2*(np.power(YFFT,2))
        else:
            X = np.fft.fft(data[int(i*window):] * hanningWindow[:int(len(data)-i*window)], nFFT)
            SSB = X[:int(nFFT/2)]
            SSB[1:] *= 2
            YFFT = np.abs(SSB/(nFFT/2))
            fftPower = 2*(np.power(YFFT,2))
        PowerChannel[:,i] = fftPower
        TimeRange[i] = time[int(i*window)]
        
    return {"Time": TimeRange, "Frequency": Frequency, "Power": PowerChannel}
        

