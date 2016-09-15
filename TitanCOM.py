# -*- coding: mbcs -*-
# Created by makepy.py version 0.5.01
# By python version 2.7.5 (default, May 15 2013, 22:43:36) [MSC v.1500 32 bit (Intel)]
# From type library 'StdScript.dll'
# On Wed Jul 16 12:15:58 2014
'TEM Scripting'
makepy_version = '0.5.01'
python_version = 0x20705f0

import win32com.client.CLSIDToClass, pythoncom, pywintypes
import win32com.client.util
from pywintypes import IID
from win32com.client import Dispatch

# The following 3 lines may need tweaking for the particular server
# Candidates are pythoncom.Missing, .Empty and .ArgNotFound
defaultNamedOptArg=pythoncom.Empty
defaultNamedNotOptArg=pythoncom.Empty
defaultUnnamedArg=pythoncom.Empty

CLSID = IID('{BC0A2B03-10FF-11D3-AE00-00A024CBA50C}')
MajorVersion = 1
MinorVersion = 9
LibraryFlags = 8
LCID = 0x0

class constants:
	AcqExposureMode_None          =0          # from enum AcqExposureMode
	AcqExposureMode_PreExposure   =2          # from enum AcqExposureMode
	AcqExposureMode_PreExposurePause=3          # from enum AcqExposureMode
	AcqExposureMode_Simultaneous  =1          # from enum AcqExposureMode
	AcqImageCorrection_Default    =1          # from enum AcqImageCorrection
	AcqImageCorrection_Unprocessed=0          # from enum AcqImageCorrection
	AcqImageFileFormat_JPG        =1          # from enum AcqImageFileFormat
	AcqImageFileFormat_PNG        =2          # from enum AcqImageFileFormat
	AcqImageFileFormat_TIFF       =0          # from enum AcqImageFileFormat
	AcqImageSize_Full             =0          # from enum AcqImageSize
	AcqImageSize_Half             =1          # from enum AcqImageSize
	AcqImageSize_Quarter          =2          # from enum AcqImageSize
	AcqShutterMode_Both           =2          # from enum AcqShutterMode
	AcqShutterMode_PostSpecimen   =1          # from enum AcqShutterMode
	AcqShutterMode_PreSpecimen    =0          # from enum AcqShutterMode
	ApertureType_Biprism          =2          # from enum ApertureType
	ApertureType_Circular         =1          # from enum ApertureType
	ApertureType_EnergySlit       =3          # from enum ApertureType
	ApertureType_FaradayCup       =4          # from enum ApertureType
	ApertureType_Unknown          =0          # from enum ApertureType
	CassetteSlotStatus_Empty      =2          # from enum CassetteSlotStatus
	CassetteSlotStatus_Error      =3          # from enum CassetteSlotStatus
	CassetteSlotStatus_Occupied   =1          # from enum CassetteSlotStatus
	CassetteSlotStatus_Unknown    =0          # from enum CassetteSlotStatus
	cmParallelIllumination        =0          # from enum CondenserMode
	cmProbeIllumination           =1          # from enum CondenserMode
	dfCartesian                   =2          # from enum DarkFieldMode
	dfConical                     =3          # from enum DarkFieldMode
	dfOff                         =1          # from enum DarkFieldMode
	plGaugePressurelevelHigh      =4          # from enum GaugePressureLevel
	plGaugePressurelevelLow       =1          # from enum GaugePressureLevel
	plGaugePressurelevelLowMedium =2          # from enum GaugePressureLevel
	plGaugePressurelevelMediumHigh=3          # from enum GaugePressureLevel
	plGaugePressurelevelUndefined =0          # from enum GaugePressureLevel
	gsInvalid                     =3          # from enum GaugeStatus
	gsOverflow                    =2          # from enum GaugeStatus
	gsUndefined                   =0          # from enum GaugeStatus
	gsUnderflow                   =1          # from enum GaugeStatus
	gsValid                       =4          # from enum GaugeStatus
	htDisabled                    =1          # from enum HightensionState
	htOff                         =2          # from enum HightensionState
	htOn                          =3          # from enum HightensionState
	imMicroProbe                  =1          # from enum IlluminationMode
	imNanoProbe                   =0          # from enum IlluminationMode
	nmAll                         =6          # from enum IlluminationNormalization
	nmCondenser                   =3          # from enum IlluminationNormalization
	nmIntensity                   =2          # from enum IlluminationNormalization
	nmMiniCondenser               =4          # from enum IlluminationNormalization
	nmObjectivePole               =5          # from enum IlluminationNormalization
	nmSpotsize                    =1          # from enum IlluminationNormalization
	InstrumentMode_STEM           =1          # from enum InstrumentMode
	InstrumentMode_TEM            =0          # from enum InstrumentMode
	lpEFTEM                       =2          # from enum LensProg
	lpRegular                     =1          # from enum LensProg
	MeasurementUnitType_Meters    =1          # from enum MeasurementUnitType
	MeasurementUnitType_Radians   =2          # from enum MeasurementUnitType
	MeasurementUnitType_Unknown   =0          # from enum MeasurementUnitType
	MechanismId_C1                =1          # from enum MechanismId
	MechanismId_C2                =2          # from enum MechanismId
	MechanismId_C3                =3          # from enum MechanismId
	MechanismId_OBJ               =4          # from enum MechanismId
	MechanismId_SA                =5          # from enum MechanismId
	MechanismId_Unknown           =0          # from enum MechanismId
	MechanismState_Aligning       =6          # from enum MechanismState
	MechanismState_Arbitrary      =4          # from enum MechanismState
	MechanismState_Disabled       =0          # from enum MechanismState
	MechanismState_Error          =7          # from enum MechanismState
	MechanismState_Homing         =5          # from enum MechanismState
	MechanismState_Inserted       =1          # from enum MechanismState
	MechanismState_Moving         =2          # from enum MechanismState
	MechanismState_Retracted      =3          # from enum MechanismState
	dtDDMMYY                      =1          # from enum PlateLabelDateFormat
	dtMMDDYY                      =2          # from enum PlateLabelDateFormat
	dtNoDate                      =0          # from enum PlateLabelDateFormat
	dtYYMMDD                      =3          # from enum PlateLabelDateFormat
	ProductFamily_Tecnai          =0          # from enum ProductFamily
	ProductFamily_Titan           =1          # from enum ProductFamily
	pdsmAlignment                 =3          # from enum ProjDetectorShiftMode
	pdsmAutoIgnore                =1          # from enum ProjDetectorShiftMode
	pdsmManual                    =2          # from enum ProjDetectorShiftMode
	pdsNearAxis                   =1          # from enum ProjectionDetectorShift
	pdsOffAxis                    =2          # from enum ProjectionDetectorShift
	pdsOnAxis                     =0          # from enum ProjectionDetectorShift
	pmDiffraction                 =2          # from enum ProjectionMode
	pmImaging                     =1          # from enum ProjectionMode
	pnmAll                        =12         # from enum ProjectionNormalization
	pnmObjective                  =10         # from enum ProjectionNormalization
	pnmProjector                  =11         # from enum ProjectionNormalization
	psmD                          =6          # from enum ProjectionSubMode
	psmLAD                        =5          # from enum ProjectionSubMode
	psmLM                         =1          # from enum ProjectionSubMode
	psmMh                         =4          # from enum ProjectionSubMode
	psmMi                         =2          # from enum ProjectionSubMode
	psmSA                         =3          # from enum ProjectionSubMode
	RefrigerantLevel_AutoloaderDewar=0          # from enum RefrigerantLevel
	RefrigerantLevel_ColumnDewar  =1          # from enum RefrigerantLevel
	RefrigerantLevel_HeliumDewar  =2          # from enum RefrigerantLevel
	spDown                        =3          # from enum ScreenPosition
	spUnknown                     =1          # from enum ScreenPosition
	spUp                          =2          # from enum ScreenPosition
	axisA                         =8          # from enum StageAxes
	axisB                         =16         # from enum StageAxes
	axisX                         =1          # from enum StageAxes
	axisXY                        =3          # from enum StageAxes
	axisY                         =2          # from enum StageAxes
	axisZ                         =4          # from enum StageAxes
	hoDoubleTilt                  =2          # from enum StageHolderType
	hoDualAxis                    =6          # from enum StageHolderType
	hoInvalid                     =4          # from enum StageHolderType
	hoNone                        =0          # from enum StageHolderType
	hoPolara                      =5          # from enum StageHolderType
	hoRotationAxis                =7          # from enum StageHolderType
	hoSingleTilt                  =1          # from enum StageHolderType
	stDisabled                    =1          # from enum StageStatus
	stGoing                       =3          # from enum StageStatus
	stMoving                      =4          # from enum StageStatus
	stNotReady                    =2          # from enum StageStatus
	stReady                       =0          # from enum StageStatus
	stWobbling                    =5          # from enum StageStatus
	E_NOT_IMPLEMENTED             =-2147155972 # from enum TEMScriptingError
	E_NOT_OK                      =-2147155969 # from enum TEMScriptingError
	E_OUT_OF_RANGE                =-2147155971 # from enum TEMScriptingError
	E_VALUE_CLIP                  =-2147155970 # from enum TEMScriptingError
	vsBusy                        =4          # from enum VacuumStatus
	vsCameraAir                   =3          # from enum VacuumStatus
	vsElse                        =6          # from enum VacuumStatus
	vsOff                         =2          # from enum VacuumStatus
	vsReady                       =5          # from enum VacuumStatus
	vsUnknown                     =1          # from enum VacuumStatus

from win32com.client import DispatchBaseClass
class AcqImage(DispatchBaseClass):
	'AcqImage Interface'
	CLSID = IID('{E15F4810-43C6-489A-9E8A-588B0949E153}')
	coclass_clsid = None

	def AsFile(self, fileName=defaultNamedNotOptArg, imageFormat=defaultNamedNotOptArg, bNormalize=False):
		'property AsFile'
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), ((8, 1), (3, 1), (11, 49)),fileName
			, imageFormat, bNormalize)

	_prop_map_get_ = {
		"AsSafeArray": (5, 2, (8195, 0), (), "AsSafeArray", None),
		"AsVariant": (6, 2, (12, 0), (), "AsVariant", None),
		"Depth": (4, 2, (3, 0), (), "Depth", None),
		"Height": (3, 2, (3, 0), (), "Height", None),
		"Name": (1, 2, (8, 0), (), "Name", None),
		"Width": (2, 2, (3, 0), (), "Width", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class AcqImages(DispatchBaseClass):
	'AcqImages Interface'
	CLSID = IID('{86365241-4D38-4642-B024-CF450CEB250B}')
	coclass_clsid = None

	# Result is of type AcqImage
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{E15F4810-43C6-489A-9E8A-588B0949E153}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{E15F4810-43C6-489A-9E8A-588B0949E153}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{E15F4810-43C6-489A-9E8A-588B0949E153}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Acquisition(DispatchBaseClass):
	'Acquisition Interface'
	CLSID = IID('{D6BBF89C-22B8-468F-80A1-947EA89269CE}')
	coclass_clsid = None

	# Result is of type AcqImages
	def AcquireImages(self):
		'method AcquireImages'
		ret = self._oleobj_.InvokeTypes(6, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, u'AcquireImages', '{86365241-4D38-4642-B024-CF450CEB250B}')
		return ret

	def AddAcqDevice(self, pDevice=defaultNamedNotOptArg):
		'method AddAcqDevice'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((9, 1),),pDevice
			)

	def AddAcqDeviceByName(self, deviceName=defaultNamedNotOptArg):
		'method AddAcqDeviceByName'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), ((8, 1),),deviceName
			)

	def RemoveAcqDevice(self, pDevice=defaultNamedNotOptArg):
		'method RemoveAcqDevice'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((9, 1),),pDevice
			)

	def RemoveAcqDeviceByName(self, deviceName=defaultNamedNotOptArg):
		'method RemoveAcqDeviceByName'
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), ((8, 1),),deviceName
			)

	def RemoveAllAcqDevices(self):
		'method RemoveAllAcqDevices'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		# Method 'Cameras' returns object of type 'CCDCameras'
		"Cameras": (10, 2, (9, 0), (), "Cameras", '{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}'),
		# Method 'Detectors' returns object of type 'STEMDetectors'
		"Detectors": (11, 2, (9, 0), (), "Detectors", '{35A2675D-E67B-4834-8940-85E7833C61A6}'),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Aperture(DispatchBaseClass):
	'Aperture Interface'
	CLSID = IID('{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Diameter": (3, 2, (5, 0), (), "Diameter", None),
		"Name": (1, 2, (8, 0), (), "Name", None),
		"Type": (2, 2, (3, 0), (), "Type", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class ApertureCollection(DispatchBaseClass):
	'ApertureCollection Interface'
	CLSID = IID('{DD08F876-9FBB-4235-A47E-FF26ADA00900}')
	coclass_clsid = None

	# Result is of type Aperture
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class ApertureMechanism(DispatchBaseClass):
	'ApertureMechanism Interface'
	CLSID = IID('{86C13CE3-934E-47DE-A211-2009E10E1EE1}')
	coclass_clsid = None

	def Disable(self):
		'Disables the mechanism'
		return self._oleobj_.InvokeTypes(9, LCID, 1, (24, 0), (),)

	def Enable(self):
		'Enables the mechanism'
		return self._oleobj_.InvokeTypes(8, LCID, 1, (24, 0), (),)

	def Retract(self):
		"retracts mechanism when it's retractable"
		return self._oleobj_.InvokeTypes(7, LCID, 1, (24, 0), (),)

	def SelectAperture(self, pVal=defaultNamedNotOptArg):
		'selects hole'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((9, 1),),pVal
			)

	_prop_map_get_ = {
		# Method 'ApertureCollection' returns object of type 'ApertureCollection'
		"ApertureCollection": (1, 2, (9, 0), (), "ApertureCollection", '{DD08F876-9FBB-4235-A47E-FF26ADA00900}'),
		"Id": (2, 2, (3, 0), (), "Id", None),
		"IsRetractable": (6, 2, (11, 0), (), "IsRetractable", None),
		# Method 'SelectedAperture' returns object of type 'Aperture'
		"SelectedAperture": (4, 2, (9, 0), (), "SelectedAperture", '{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}'),
		"State": (5, 2, (3, 0), (), "State", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class ApertureMechanismCollection(DispatchBaseClass):
	'ApertureMechanismCollection Interface'
	CLSID = IID('{A5015A2D-0436-472E-9140-43DC805B1EEE}')
	coclass_clsid = None

	# Result is of type ApertureMechanism
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{86C13CE3-934E-47DE-A211-2009E10E1EE1}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{86C13CE3-934E-47DE-A211-2009E10E1EE1}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{86C13CE3-934E-47DE-A211-2009E10E1EE1}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class AutoLoader(DispatchBaseClass):
	'AutoLoader Interface'
	CLSID = IID('{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}')
	coclass_clsid = None

	def BufferCycle(self):
		'Perform buffer cycle'
		return self._oleobj_.InvokeTypes(4, LCID, 1, (24, 0), (),)

	def LoadCartridge(self, fromSlot=defaultNamedNotOptArg):
		'Load cartridge from a cassette'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((3, 1),),fromSlot
			)

	def PerformCassetteInventory(self):
		'Perform cassette inventory'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), (),)

	# The method SlotStatus is actually a property, but must be used as a method to correctly pass the arguments
	def SlotStatus(self, slot=defaultNamedNotOptArg):
		'Slot status'
		return self._oleobj_.InvokeTypes(12, LCID, 2, (3, 0), ((3, 1),),slot
			)

	def UnloadCartridge(self):
		'Unload cartridge back to the cassette'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"AutoLoaderAvailable": (10, 2, (11, 0), (), "AutoLoaderAvailable", None),
		"NumberOfCassetteSlots": (11, 2, (3, 0), (), "NumberOfCassetteSlots", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class BlankerShutter(DispatchBaseClass):
	'BlankerShutter Interface'
	CLSID = IID('{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}')
	coclass_clsid = None

	_prop_map_get_ = {
		"ShutterOverrideOn": (10, 2, (11, 0), (), "ShutterOverrideOn", None),
	}
	_prop_map_put_ = {
		"ShutterOverrideOn": ((10, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class CCDAcqParams(DispatchBaseClass):
	'CCDAcqParams Interface'
	CLSID = IID('{C03DB779-1345-42AB-9304-95B85789163D}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Binning": (3, 2, (3, 0), (), "Binning", None),
		"ExposureMode": (5, 2, (3, 0), (), "ExposureMode", None),
		"ExposureTime": (2, 2, (5, 0), (), "ExposureTime", None),
		"ImageCorrection": (4, 2, (3, 0), (), "ImageCorrection", None),
		"ImageSize": (1, 2, (3, 0), (), "ImageSize", None),
		"MaxPreExposurePauseTime": (10, 2, (5, 0), (), "MaxPreExposurePauseTime", None),
		"MaxPreExposureTime": (7, 2, (5, 0), (), "MaxPreExposureTime", None),
		"MinPreExposurePauseTime": (9, 2, (5, 0), (), "MinPreExposurePauseTime", None),
		"MinPreExposureTime": (6, 2, (5, 0), (), "MinPreExposureTime", None),
		"PreExposurePauseTime": (11, 2, (5, 0), (), "PreExposurePauseTime", None),
		"PreExposureTime": (8, 2, (5, 0), (), "PreExposureTime", None),
	}
	_prop_map_put_ = {
		"Binning": ((3, LCID, 4, 0),()),
		"ExposureMode": ((5, LCID, 4, 0),()),
		"ExposureTime": ((2, LCID, 4, 0),()),
		"ImageCorrection": ((4, LCID, 4, 0),()),
		"ImageSize": ((1, LCID, 4, 0),()),
		"PreExposurePauseTime": ((11, LCID, 4, 0),()),
		"PreExposureTime": ((8, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class CCDCamera(DispatchBaseClass):
	'CCDCamera Interface'
	CLSID = IID('{E44E1565-4131-4937-B273-78219E090845}')
	coclass_clsid = None

	_prop_map_get_ = {
		# Method 'AcqParams' returns object of type 'CCDAcqParams'
		"AcqParams": (2, 2, (9, 0), (), "AcqParams", '{C03DB779-1345-42AB-9304-95B85789163D}'),
		# Method 'Info' returns object of type 'CCDCameraInfo'
		"Info": (1, 2, (9, 0), (), "Info", '{024DED60-B124-4514-BFE2-02C0F5C51DB9}'),
	}
	_prop_map_put_ = {
		"AcqParams": ((2, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class CCDCameraInfo(DispatchBaseClass):
	'CCDCameraInfo Interface'
	CLSID = IID('{024DED60-B124-4514-BFE2-02C0F5C51DB9}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Binnings": (5, 2, (8195, 0), (), "Binnings", None),
		"BinningsAsVariant": (8, 2, (12, 0), (), "BinningsAsVariant", None),
		"Height": (3, 2, (3, 0), (), "Height", None),
		"Name": (1, 2, (8, 0), (), "Name", None),
		# Method 'PixelSize' returns object of type 'Vector'
		"PixelSize": (4, 2, (9, 0), (), "PixelSize", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"ShutterMode": (7, 2, (3, 0), (), "ShutterMode", None),
		"ShutterModes": (6, 2, (8195, 0), (), "ShutterModes", None),
		"ShutterModesAsVariant": (9, 2, (12, 0), (), "ShutterModesAsVariant", None),
		"Width": (2, 2, (3, 0), (), "Width", None),
	}
	_prop_map_put_ = {
		"ShutterMode": ((7, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class CCDCameras(DispatchBaseClass):
	'CCDCameras Interface'
	CLSID = IID('{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}')
	coclass_clsid = None

	# Result is of type CCDCamera
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{E44E1565-4131-4937-B273-78219E090845}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{E44E1565-4131-4937-B273-78219E090845}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{E44E1565-4131-4937-B273-78219E090845}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Camera(DispatchBaseClass):
	'Interface to the camera system'
	CLSID = IID('{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}')
	coclass_clsid = None

	def TakeExposure(self):
		'Take a photo (uses current parameter settings)'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"ExposureNumber": (18, 2, (3, 0), (), "ExposureNumber", None),
		"FilmText": (14, 2, (8, 0), (), "FilmText", None),
		"IsSmallScreenDown": (12, 2, (11, 0), (), "IsSmallScreenDown", None),
		"MainScreen": (11, 2, (3, 0), (), "MainScreen", None),
		"ManualExposure": (21, 2, (11, 0), (), "ManualExposure", None),
		"ManualExposureTime": (15, 2, (5, 0), (), "ManualExposureTime", None),
		"MeasuredExposureTime": (13, 2, (5, 0), (), "MeasuredExposureTime", None),
		"PlateLabelDateType": (22, 2, (3, 0), (), "PlateLabelDateType", None),
		"PlateuMarker": (17, 2, (11, 0), (), "PlateuMarker", None),
		"ScreenCurrent": (25, 2, (5, 0), (), "ScreenCurrent", None),
		"ScreenDim": (23, 2, (11, 0), (), "ScreenDim", None),
		"ScreenDimText": (24, 2, (8, 0), (), "ScreenDimText", None),
		"Stock": (10, 2, (3, 0), (), "Stock", None),
		"Usercode": (19, 2, (8, 0), (), "Usercode", None),
	}
	_prop_map_put_ = {
		"ExposureNumber": ((18, LCID, 4, 0),()),
		"FilmText": ((14, LCID, 4, 0),()),
		"MainScreen": ((11, LCID, 4, 0),()),
		"ManualExposure": ((21, LCID, 4, 0),()),
		"ManualExposureTime": ((15, LCID, 4, 0),()),
		"PlateLabelDateType": ((22, LCID, 4, 0),()),
		"PlateuMarker": ((17, LCID, 4, 0),()),
		"ScreenDim": ((23, LCID, 4, 0),()),
		"ScreenDimText": ((24, LCID, 4, 0),()),
		"Usercode": ((19, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Configuration(DispatchBaseClass):
	'Configuration Interface'
	CLSID = IID('{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}')
	coclass_clsid = None

	_prop_map_get_ = {
		"ProductFamily": (1, 2, (3, 0), (), "ProductFamily", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Gauge(DispatchBaseClass):
	'Utility object: Vacuum system gauge data (pressure)'
	CLSID = IID('{52020820-18BF-11D3-86E1-00C04FC126DD}')
	coclass_clsid = None

	def Read(self):
		'Read gauge settings'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"Name": (10, 2, (8, 0), (), "Name", None),
		"Pressure": (11, 2, (5, 0), (), "Pressure", None),
		"PressureLevel": (13, 2, (3, 0), (), "PressureLevel", None),
		"Status": (12, 2, (3, 0), (), "Status", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Gauges(DispatchBaseClass):
	'Vacuum system gauges collection'
	CLSID = IID('{6E6F03B0-2ECE-11D3-AE79-004095005B07}')
	coclass_clsid = None

	# Result is of type Gauge
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'Get individual gauge'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 0),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{52020820-18BF-11D3-86E1-00C04FC126DD}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'Get individual gauge'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 0),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{52020820-18BF-11D3-86E1-00C04FC126DD}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{52020820-18BF-11D3-86E1-00C04FC126DD}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Gun(DispatchBaseClass):
	'Gun Interface'
	CLSID = IID('{E6F00870-3164-11D3-B4C8-00A024CB9221}')
	coclass_clsid = None

	_prop_map_get_ = {
		"HTMaxValue": (12, 2, (5, 0), (), "HTMaxValue", None),
		"HTState": (10, 2, (3, 0), (), "HTState", None),
		"HTValue": (11, 2, (5, 0), (), "HTValue", None),
		# Method 'Shift' returns object of type 'Vector'
		"Shift": (13, 2, (9, 0), (), "Shift", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		# Method 'Tilt' returns object of type 'Vector'
		"Tilt": (14, 2, (9, 0), (), "Tilt", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
	}
	_prop_map_put_ = {
		"HTState": ((10, LCID, 4, 0),()),
		"HTValue": ((11, LCID, 4, 0),()),
		"Shift": ((13, LCID, 4, 0),()),
		"Tilt": ((14, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IUserButton(DispatchBaseClass):
	'User button'
	CLSID = IID('{E6F00871-3164-11D3-B4C8-00A024CB9221}')
	coclass_clsid = IID('{3A4CE1F0-3A05-11D3-AE81-004095005B07}')

	_prop_map_get_ = {
		"Assignment": (12, 2, (8, 0), (), "Assignment", None),
		"Label": (11, 2, (8, 0), (), "Label", None),
		"Name": (10, 2, (8, 0), (), "Name", None),
	}
	_prop_map_put_ = {
		"Assignment": ((12, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Illumination(DispatchBaseClass):
	'Illumination Interface'
	CLSID = IID('{EF960690-1C38-11D3-AE0B-00A024CBA50C}')
	coclass_clsid = None

	def Normalize(self, nm=defaultNamedNotOptArg):
		'Normalization of illumination system'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((3, 0),),nm
			)

	_prop_map_get_ = {
		"BeamBlanked": (16, 2, (11, 0), (), "BeamBlanked", None),
		"CondenserMode": (22, 2, (3, 0), (), "CondenserMode", None),
		# Method 'CondenserStigmator' returns object of type 'Vector'
		"CondenserStigmator": (20, 2, (9, 0), (), "CondenserStigmator", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"ConvergenceAngle": (25, 2, (5, 0), (), "ConvergenceAngle", None),
		"DFMode": (21, 2, (3, 0), (), "DFMode", None),
		"IlluminatedArea": (23, 2, (5, 0), (), "IlluminatedArea", None),
		"Intensity": (13, 2, (5, 0), (), "Intensity", None),
		"IntensityLimitEnabled": (15, 2, (11, 0), (), "IntensityLimitEnabled", None),
		"IntensityZoomEnabled": (14, 2, (11, 0), (), "IntensityZoomEnabled", None),
		"Mode": (11, 2, (3, 0), (), "Mode", None),
		"ProbeDefocus": (24, 2, (5, 0), (), "ProbeDefocus", None),
		# Method 'RotationCenter' returns object of type 'Vector'
		"RotationCenter": (19, 2, (9, 0), (), "RotationCenter", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		# Method 'Shift' returns object of type 'Vector'
		"Shift": (17, 2, (9, 0), (), "Shift", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"SpotsizeIndex": (12, 2, (3, 0), (), "SpotsizeIndex", None),
		# Method 'StemFullScanFieldOfView' returns object of type 'Vector'
		"StemFullScanFieldOfView": (28, 2, (9, 0), (), "StemFullScanFieldOfView", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"StemMagnification": (26, 2, (5, 0), (), "StemMagnification", None),
		"StemRotation": (27, 2, (5, 0), (), "StemRotation", None),
		# Method 'Tilt' returns object of type 'Vector'
		"Tilt": (18, 2, (9, 0), (), "Tilt", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
	}
	_prop_map_put_ = {
		"BeamBlanked": ((16, LCID, 4, 0),()),
		"CondenserMode": ((22, LCID, 4, 0),()),
		"CondenserStigmator": ((20, LCID, 4, 0),()),
		"DFMode": ((21, LCID, 4, 0),()),
		"IlluminatedArea": ((23, LCID, 4, 0),()),
		"Intensity": ((13, LCID, 4, 0),()),
		"IntensityLimitEnabled": ((15, LCID, 4, 0),()),
		"IntensityZoomEnabled": ((14, LCID, 4, 0),()),
		"Mode": ((11, LCID, 4, 0),()),
		"RotationCenter": ((19, LCID, 4, 0),()),
		"Shift": ((17, LCID, 4, 0),()),
		"SpotsizeIndex": ((12, LCID, 4, 0),()),
		"StemMagnification": ((26, LCID, 4, 0),()),
		"StemRotation": ((27, LCID, 4, 0),()),
		"Tilt": ((18, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class InstrumentInterface(DispatchBaseClass):
	'Instrument Interface'
	CLSID = IID('{BC0A2B11-10FF-11D3-AE00-00A024CBA50C}')
	coclass_clsid = IID('{02CDC9A1-1F1D-11D3-AE11-00A024CBA50C}')

	def NormalizeAll(self):
		'Normalize all lenses'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), (),)

	def ReturnError(self, TE=defaultNamedNotOptArg):
		'Dummy to test error codes'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((3, 1),),TE
			)

	_prop_map_get_ = {
		# Method 'Acquisition' returns object of type 'Acquisition'
		"Acquisition": (31, 2, (9, 0), (), "Acquisition", '{D6BBF89C-22B8-468F-80A1-947EA89269CE}'),
		# Method 'ApertureMechanismCollection' returns object of type 'ApertureMechanismCollection'
		"ApertureMechanismCollection": (33, 2, (9, 0), (), "ApertureMechanismCollection", '{A5015A2D-0436-472E-9140-43DC805B1EEE}'),
		# Method 'AutoLoader' returns object of type 'AutoLoader'
		"AutoLoader": (27, 2, (9, 0), (), "AutoLoader", '{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}'),
		"AutoNormalizeEnabled": (2, 2, (11, 0), (), "AutoNormalizeEnabled", None),
		# Method 'BlankerShutter' returns object of type 'BlankerShutter'
		"BlankerShutter": (29, 2, (9, 0), (), "BlankerShutter", '{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}'),
		# Method 'Camera' returns object of type 'Camera'
		"Camera": (21, 2, (9, 0), (), "Camera", '{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}'),
		# Method 'Configuration' returns object of type 'Configuration'
		"Configuration": (32, 2, (9, 0), (), "Configuration", '{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}'),
		# Method 'Gun' returns object of type 'Gun'
		"Gun": (25, 2, (9, 0), (), "Gun", '{E6F00870-3164-11D3-B4C8-00A024CB9221}'),
		# Method 'Illumination' returns object of type 'Illumination'
		"Illumination": (23, 2, (9, 0), (), "Illumination", '{EF960690-1C38-11D3-AE0B-00A024CBA50C}'),
		# Method 'InstrumentModeControl' returns object of type 'InstrumentModeControl'
		"InstrumentModeControl": (30, 2, (9, 0), (), "InstrumentModeControl", '{8DC0FC71-FF15-40D8-8174-092218D8B76B}'),
		# Method 'Projection' returns object of type 'Projection'
		"Projection": (24, 2, (9, 0), (), "Projection", '{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}'),
		# Method 'Stage' returns object of type 'Stage'
		"Stage": (22, 2, (9, 0), (), "Stage", '{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}'),
		"StagePosition": (12, 2, (9, 0), (), "StagePosition", None),
		# Method 'TemperatureControl' returns object of type 'TemperatureControl'
		"TemperatureControl": (28, 2, (9, 0), (), "TemperatureControl", '{71B6E709-B21F-435F-9529-1AEE55CFA029}'),
		# Method 'UserButtons' returns object of type 'UserButtons'
		"UserButtons": (26, 2, (9, 0), (), "UserButtons", '{50C21D10-317F-11D3-B4C8-00A024CB9221}'),
		# Method 'Vacuum' returns object of type 'Vacuum'
		"Vacuum": (20, 2, (9, 0), (), "Vacuum", '{C7646442-1115-11D3-AE00-00A024CBA50C}'),
		"Vector": (11, 2, (9, 0), (), "Vector", None),
	}
	_prop_map_put_ = {
		"AutoNormalizeEnabled": ((2, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class InstrumentModeControl(DispatchBaseClass):
	'InstrumentModeControl Interface'
	CLSID = IID('{8DC0FC71-FF15-40D8-8174-092218D8B76B}')
	coclass_clsid = None

	_prop_map_get_ = {
		"InstrumentMode": (11, 2, (3, 0), (), "InstrumentMode", None),
		"StemAvailable": (10, 2, (11, 0), (), "StemAvailable", None),
	}
	_prop_map_put_ = {
		"InstrumentMode": ((11, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Projection(DispatchBaseClass):
	'Projection Interface'
	CLSID = IID('{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}')
	coclass_clsid = None

	def ChangeProjectionIndex(self, addVal=defaultNamedNotOptArg):
		'Change the currently available projection index'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((3, 1),),addVal
			)

	def Normalize(self, norm=defaultNamedNotOptArg):
		'Normalize lenses of projection system'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), ((3, 1),),norm
			)

	def ResetDefocus(self):
		'Reset Defocus'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"CameraLength": (13, 2, (5, 0), (), "CameraLength", None),
		"CameraLengthIndex": (15, 2, (3, 0), (), "CameraLengthIndex", None),
		"Defocus": (21, 2, (5, 0), (), "Defocus", None),
		"DetectorShift": (30, 2, (3, 0), (), "DetectorShift", None),
		"DetectorShiftMode": (31, 2, (3, 0), (), "DetectorShiftMode", None),
		# Method 'DiffractionShift' returns object of type 'Vector'
		"DiffractionShift": (18, 2, (9, 0), (), "DiffractionShift", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		# Method 'DiffractionStigmator' returns object of type 'Vector'
		"DiffractionStigmator": (19, 2, (9, 0), (), "DiffractionStigmator", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"Focus": (11, 2, (5, 0), (), "Focus", None),
		# Method 'ImageBeamShift' returns object of type 'Vector'
		"ImageBeamShift": (17, 2, (9, 0), (), "ImageBeamShift", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		# Method 'ImageBeamTilt' returns object of type 'Vector'
		"ImageBeamTilt": (32, 2, (9, 0), (), "ImageBeamTilt", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"ImageRotation": (29, 2, (5, 0), (), "ImageRotation", None),
		# Method 'ImageShift' returns object of type 'Vector'
		"ImageShift": (16, 2, (9, 0), (), "ImageShift", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"LensProgram": (28, 2, (3, 0), (), "LensProgram", None),
		"Magnification": (12, 2, (5, 0), (), "Magnification", None),
		"MagnificationIndex": (14, 2, (3, 0), (), "MagnificationIndex", None),
		"Mode": (10, 2, (3, 0), (), "Mode", None),
		"ObjectiveExcitation": (26, 2, (5, 0), (), "ObjectiveExcitation", None),
		# Method 'ObjectiveStigmator' returns object of type 'Vector'
		"ObjectiveStigmator": (20, 2, (9, 0), (), "ObjectiveStigmator", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
		"ProjectionIndex": (27, 2, (3, 0), (), "ProjectionIndex", None),
		"SubMode": (23, 2, (3, 0), (), "SubMode", None),
		"SubModeMaxIndex": (25, 2, (3, 0), (), "SubModeMaxIndex", None),
		"SubModeMinIndex": (24, 2, (3, 0), (), "SubModeMinIndex", None),
		"SubModeString": (22, 2, (8, 0), (), "SubModeString", None),tem.t
	}
	_prop_map_put_ = {
		"CameraLengthIndex": ((15, LCID, 4, 0),()),
		"Defocus": ((21, LCID, 4, 0),()),
		"DetectorShift": ((30, LCID, 4, 0),()),
		"DetectorShiftMode": ((31, LCID, 4, 0),()),
		"DiffractionShift": ((18, LCID, 4, 0),()),
		"DiffractionStigmator": ((19, LCID, 4, 0),()),
		"Focus": ((11, LCID, 4, 0),()),
		"ImageBeamShift": ((17, LCID, 4, 0),()),
		"ImageBeamTilt": ((32, LCID, 4, 0),()),
		"ImageShift": ((16, LCID, 4, 0),()),
		"LensProgram": ((28, LCID, 4, 0),()),
		"MagnificationIndex": ((14, LCID, 4, 0),()),
		"Mode": ((10, LCID, 4, 0),()),
		"ObjectiveStigmator": ((20, LCID, 4, 0),()),
		"ProjectionIndex": ((27, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class STEMAcqParams(DispatchBaseClass):
	'STEMAcqParams Interface'
	CLSID = IID('{DDC14710-6152-4963-AEA4-C67BA784C6B4}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Binning": (3, 2, (3, 0), (), "Binning", None),
		"DwellTime": (2, 2, (5, 0), (), "DwellTime", None),
		"ImageSize": (1, 2, (3, 0), (), "ImageSize", None),
		# Method 'MaxResolution' returns object of type 'Vector'
		"MaxResolution": (4, 2, (9, 0), (), "MaxResolution", '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}'),
	}
	_prop_map_put_ = {
		"Binning": ((3, LCID, 4, 0),()),
		"DwellTime": ((2, LCID, 4, 0),()),
		"ImageSize": ((1, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class STEMDetector(DispatchBaseClass):
	'STEMDetector Interface'
	CLSID = IID('{D77C0D65-A1DD-4D0A-AF25-C280046A5719}')
	coclass_clsid = None

	_prop_map_get_ = {
		# Method 'Info' returns object of type 'STEMDetectorInfo'
		"Info": (1, 2, (9, 0), (), "Info", '{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}'),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class STEMDetectorInfo(DispatchBaseClass):
	'STEMDetectorInfo Interface'
	CLSID = IID('{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}')
	coclass_clsid = None

	_prop_map_get_ = {
		"Binnings": (4, 2, (8195, 0), (), "Binnings", None),
		"BinningsAsVariant": (5, 2, (12, 0), (), "BinningsAsVariant", None),
		"Brightness": (2, 2, (5, 0), (), "Brightness", None),
		"Contrast": (3, 2, (5, 0), (), "Contrast", None),
		"Name": (1, 2, (8, 0), (), "Name", None),
	}
	_prop_map_put_ = {
		"Brightness": ((2, LCID, 4, 0),()),
		"Contrast": ((3, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class STEMDetectors(DispatchBaseClass):
	'STEMDetectors Interface'
	CLSID = IID('{35A2675D-E67B-4834-8940-85E7833C61A6}')
	coclass_clsid = None

	# Result is of type STEMDetector
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{D77C0D65-A1DD-4D0A-AF25-C280046A5719}')
		return ret

	_prop_map_get_ = {
		# Method 'AcqParams' returns object of type 'STEMAcqParams'
		"AcqParams": (2, 2, (9, 0), (), "AcqParams", '{DDC14710-6152-4963-AEA4-C67BA784C6B4}'),
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
		"AcqParams": ((2, LCID, 4, 0),()),
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'property Item'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 1),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{D77C0D65-A1DD-4D0A-AF25-C280046A5719}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{D77C0D65-A1DD-4D0A-AF25-C280046A5719}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Stage(DispatchBaseClass):
	'Stage Interface'
	CLSID = IID('{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}')
	coclass_clsid = None

	# Result is of type StageAxisData
	# The method AxisData is actually a property, but must be used as a method to correctly pass the arguments
	def AxisData(self, mask=defaultNamedNotOptArg):
		'Axis data'
		ret = self._oleobj_.InvokeTypes(13, LCID, 2, (9, 0), ((3, 0),),mask
			)
		if ret is not None:
			ret = Dispatch(ret, u'AxisData', '{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}')
		return ret

	def Goto(self, newPos=defaultNamedNotOptArg, mask=defaultNamedNotOptArg):
		'Goto a position'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((9, 0), (3, 0)),newPos
			, mask)

	def GotoWithSpeed(self, newPos=defaultNamedNotOptArg, mask=defaultNamedNotOptArg, speed=defaultNamedNotOptArg):
		'Goto position with speed'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), ((9, 1), (3, 1), (5, 1)),newPos
			, mask, speed)

	def MoveTo(self, newPos=defaultNamedNotOptArg, mask=defaultNamedNotOptArg):
		'Move to a position'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), ((9, 0), (3, 0)),newPos
			, mask)

	_prop_map_get_ = {
		"Holder": (12, 2, (3, 0), (), "Holder", None),
		# Method 'Position' returns object of type 'StagePosition'
		"Position": (11, 2, (9, 0), (), "Position", '{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}'),
		"Status": (10, 2, (3, 0), (), "Status", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class StageAxisData(DispatchBaseClass):
	'Utility object: axis data'
	CLSID = IID('{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}')
	coclass_clsid = None

	_prop_map_get_ = {
		"MaxPos": (2, 2, (5, 0), (), "MaxPos", None),
		"MinPos": (1, 2, (5, 0), (), "MinPos", None),
		"UnitType": (3, 2, (3, 0), (), "UnitType", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class StagePosition(DispatchBaseClass):
	'Utility object: stage coordinates'
	CLSID = IID('{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}')
	coclass_clsid = None

	def GetAsArray(self, pos=defaultNamedNotOptArg):
		'Return position in an array'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), ((16389, 0),),pos
			)

	def SetAsArray(self, pos=defaultNamedNotOptArg):
		'Set position from an array'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (24, 0), ((16389, 0),),pos
			)

	_prop_map_get_ = {
		"A": (13, 2, (5, 0), (), "A", None),
		"B": (14, 2, (5, 0), (), "B", None),
		"X": (10, 2, (5, 0), (), "X", None),
		"Y": (11, 2, (5, 0), (), "Y", None),
		"Z": (12, 2, (5, 0), (), "Z", None),
	}
	_prop_map_put_ = {
		"A": ((13, LCID, 4, 0),()),
		"B": ((14, LCID, 4, 0),()),
		"X": ((10, LCID, 4, 0),()),
		"Y": ((11, LCID, 4, 0),()),
		"Z": ((12, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class TemperatureControl(DispatchBaseClass):
	'Temperature Control Interface'
	CLSID = IID('{71B6E709-B21F-435F-9529-1AEE55CFA029}')
	coclass_clsid = None

	def ForceRefill(self):
		'Force refrigerant refill'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (24, 0), (),)

	# The method RefrigerantLevel is actually a property, but must be used as a method to correctly pass the arguments
	def RefrigerantLevel(self, rl=defaultNamedNotOptArg):
		'Get refrigerant level'
		return self._oleobj_.InvokeTypes(11, LCID, 2, (5, 0), ((3, 1),),rl
			)

	_prop_map_get_ = {
		"DewarsAreBusyFilling": (14, 2, (11, 0), (), "DewarsAreBusyFilling", None),
		"DewarsRemainingTime": (13, 2, (3, 0), (), "DewarsRemainingTime", None),
		"TemperatureControlAvailable": (10, 2, (11, 0), (), "TemperatureControlAvailable", None),
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class UserButtonEvent:
	'Standard scripting event interface'
	CLSID = CLSID_Sink = IID('{02CDC9A2-1F1D-11D3-AE11-00A024CBA50C}')
	coclass_clsid = IID('{3A4CE1F0-3A05-11D3-AE81-004095005B07}')
	_public_methods_ = [] # For COM Server support
	_dispid_to_func_ = {
		        1 : "OnPressed",
		}

	def __init__(self, oobj = None):
		if oobj is None:
			self._olecp = None
		else:
			import win32com.server.util
			from win32com.server.policy import EventHandlerPolicy
			cpc=oobj._oleobj_.QueryInterface(pythoncom.IID_IConnectionPointContainer)
			cp=cpc.FindConnectionPoint(self.CLSID_Sink)
			cookie=cp.Advise(win32com.server.util.wrap(self, usePolicy=EventHandlerPolicy))
			self._olecp,self._olecp_cookie = cp,cookie
	def __del__(self):
		try:
			self.close()
		except pythoncom.com_error:
			pass
	def close(self):
		if self._olecp is not None:
			cp,cookie,self._olecp,self._olecp_cookie = self._olecp,self._olecp_cookie,None,None
			cp.Unadvise(cookie)
	def _query_interface_(self, iid):
		import win32com.server.util
		if iid==self.CLSID_Sink: return win32com.server.util.wrap(self)

	# Event Handlers
	# If you create handlers, they should have the following prototypes:
#	def OnPressed(self):
#		'Button pressed event'


class UserButtons(DispatchBaseClass):
	'User buttons collection'
	CLSID = IID('{50C21D10-317F-11D3-B4C8-00A024CB9221}')
	coclass_clsid = None

	# Result is of type IUserButton
	# The method Item is actually a property, but must be used as a method to correctly pass the arguments
	def Item(self, index=defaultNamedNotOptArg):
		'Get individual Button'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 0),),index
			)
		if ret is not None:
			ret = Dispatch(ret, u'Item', '{E6F00871-3164-11D3-B4C8-00A024CB9221}')
		return ret

	_prop_map_get_ = {
		"Count": (1, 2, (3, 0), (), "Count", None),
	}
	_prop_map_put_ = {
	}
	# Default method for this class is 'Item'
	def __call__(self, index=defaultNamedNotOptArg):
		'Get individual Button'
		ret = self._oleobj_.InvokeTypes(0, LCID, 2, (9, 0), ((12, 0),),index
			)
		if ret is not None:
			ret = Dispatch(ret, '__call__', '{E6F00871-3164-11D3-B4C8-00A024CB9221}')
		return ret

	def __unicode__(self, *args):
		try:
			return unicode(self.__call__(*args))
		except pythoncom.com_error:
			return repr(self)
	def __str__(self, *args):
		return str(self.__unicode__(*args))
	def __int__(self, *args):
		return int(self.__call__(*args))
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,2,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, '{E6F00871-3164-11D3-B4C8-00A024CB9221}')
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __nonzero__(self):
		return True

class Vacuum(DispatchBaseClass):
	'Vacuum System interface'
	CLSID = IID('{C7646442-1115-11D3-AE00-00A024CBA50C}')
	coclass_clsid = None

	def RunBufferCycle(self):
		'Request a buffer cycle'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"ColumnValvesOpen": (13, 2, (11, 0), (), "ColumnValvesOpen", None),
		# Method 'Gauges' returns object of type 'Gauges'
		"Gauges": (12, 2, (9, 0), (), "Gauges", '{6E6F03B0-2ECE-11D3-AE79-004095005B07}'),
		"PVPRunning": (11, 2, (11, 0), (), "PVPRunning", None),
		"Status": (10, 2, (3, 0), (), "Status", None),
	}
	_prop_map_put_ = {
		"ColumnValvesOpen": ((13, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class Vector(DispatchBaseClass):
	'Utility object: Vector'
	CLSID = IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')
	coclass_clsid = None

	_prop_map_get_ = {
		"X": (1, 2, (5, 0), (), "X", None),
		"Y": (2, 2, (5, 0), (), "Y", None),
	}
	_prop_map_put_ = {
		"X": ((1, LCID, 4, 0),()),
		"Y": ((2, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

from win32com.client import CoClassBaseClass
# This CoClass is known by the name 'TEMScripting.Instrument.1'
class Instrument(CoClassBaseClass): # A CoClass
	# Interface to access all subsystems
	CLSID = IID('{02CDC9A1-1F1D-11D3-AE11-00A024CBA50C}')
	coclass_sources = [
	]
	coclass_interfaces = [
		InstrumentInterface,
	]
	default_interface = InstrumentInterface

class UserButton(CoClassBaseClass): # A CoClass
	# TEM user buttons
	CLSID = IID('{3A4CE1F0-3A05-11D3-AE81-004095005B07}')
	coclass_sources = [
		UserButtonEvent,
	]
	default_source = UserButtonEvent
	coclass_interfaces = [
		IUserButton,
	]
	default_interface = IUserButton

AcqImage_vtables_dispatch_ = 1
AcqImage_vtables_ = [
	(( u'Name' , u'pVal' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pVal' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Depth' , u'pVal' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'AsSafeArray' , u'pVal' , ), 5, (5, (), [ (24579, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AsVariant' , u'pVal' , ), 6, (6, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'AsFile' , u'fileName' , u'imageFormat' , u'bNormalize' , ), 7, (7, (), [ 
			(8, 1, None, None) , (3, 1, None, None) , (11, 49, 'False', None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

AcqImages_vtables_dispatch_ = 1
AcqImages_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pVal' , ), 0, (0, (), [ (12, 1, None, None) , 
			(16393, 10, None, "IID('{E15F4810-43C6-489A-9E8A-588B0949E153}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

Acquisition_vtables_dispatch_ = 1
Acquisition_vtables_ = [
	(( u'AddAcqDevice' , u'pDevice' , ), 1, (1, (), [ (9, 1, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'AddAcqDeviceByName' , u'deviceName' , ), 2, (2, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'RemoveAcqDevice' , u'pDevice' , ), 3, (3, (), [ (9, 1, None, None) , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'RemoveAcqDeviceByName' , u'deviceName' , ), 4, (4, (), [ (8, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'RemoveAllAcqDevices' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AcquireImages' , u'ppImageCol' , ), 6, (6, (), [ (16393, 10, None, "IID('{86365241-4D38-4642-B024-CF450CEB250B}')") , ], 1 , 1 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Cameras' , u'pCol' , ), 10, (10, (), [ (16393, 10, None, "IID('{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}')") , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Detectors' , u'pCol' , ), 11, (11, (), [ (16393, 10, None, "IID('{35A2675D-E67B-4834-8940-85E7833C61A6}')") , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
]

Aperture_vtables_dispatch_ = 1
Aperture_vtables_ = [
	(( u'Name' , u'pVal' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Type' , u'pVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Diameter' , u'pVal' , ), 3, (3, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

ApertureCollection_vtables_dispatch_ = 1
ApertureCollection_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pVal' , ), 0, (0, (), [ (12, 1, None, None) , 
			(16393, 10, None, "IID('{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

ApertureMechanism_vtables_dispatch_ = 1
ApertureMechanism_vtables_ = [
	(( u'ApertureCollection' , u'pVal' , ), 1, (1, (), [ (16393, 10, None, "IID('{DD08F876-9FBB-4235-A47E-FF26ADA00900}')") , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Id' , u'pVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'SelectAperture' , u'pVal' , ), 3, (3, (), [ (9, 1, None, "IID('{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')") , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'SelectedAperture' , u'pVal' , ), 4, (4, (), [ (16393, 10, None, "IID('{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'State' , u'pVal' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'IsRetractable' , u'pVal' , ), 6, (6, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Retract' , ), 7, (7, (), [ ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Enable' , ), 8, (8, (), [ ], 1 , 1 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Disable' , ), 9, (9, (), [ ], 1 , 1 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
]

ApertureMechanismCollection_vtables_dispatch_ = 1
ApertureMechanismCollection_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pVal' , ), 0, (0, (), [ (12, 1, None, None) , 
			(16393, 10, None, "IID('{86C13CE3-934E-47DE-A211-2009E10E1EE1}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

AutoLoader_vtables_dispatch_ = 1
AutoLoader_vtables_ = [
	(( u'LoadCartridge' , u'fromSlot' , ), 1, (1, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'UnloadCartridge' , ), 2, (2, (), [ ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'PerformCassetteInventory' , ), 3, (3, (), [ ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'BufferCycle' , ), 4, (4, (), [ ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'AutoLoaderAvailable' , u'pAvail' , ), 10, (10, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'NumberOfCassetteSlots' , u'nrOfSlots' , ), 11, (11, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'SlotStatus' , u'slot' , u'Status' , ), 12, (12, (), [ (3, 1, None, None) , 
			(16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

BlankerShutter_vtables_dispatch_ = 1
BlankerShutter_vtables_ = [
	(( u'ShutterOverrideOn' , u'pOverride' , ), 10, (10, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'ShutterOverrideOn' , u'pOverride' , ), 10, (10, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
]

CCDAcqParams_vtables_dispatch_ = 1
CCDAcqParams_vtables_ = [
	(( u'ImageSize' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'ImageSize' , u'pVal' , ), 1, (1, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'ExposureTime' , u'pVal' , ), 2, (2, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ExposureTime' , u'pVal' , ), 2, (2, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Binning' , u'pVal' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Binning' , u'pVal' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ImageCorrection' , u'pVal' , ), 4, (4, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ImageCorrection' , u'pVal' , ), 4, (4, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'ExposureMode' , u'pVal' , ), 5, (5, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'ExposureMode' , u'pVal' , ), 5, (5, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'MinPreExposureTime' , u'minVal' , ), 6, (6, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'MaxPreExposureTime' , u'maxVal' , ), 7, (7, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'PreExposureTime' , u'pVal' , ), 8, (8, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'PreExposureTime' , u'pVal' , ), 8, (8, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'MinPreExposurePauseTime' , u'minVal' , ), 9, (9, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'MaxPreExposurePauseTime' , u'maxVal' , ), 10, (10, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'PreExposurePauseTime' , u'pVal' , ), 11, (11, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'PreExposurePauseTime' , u'pVal' , ), 11, (11, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
]

CCDCamera_vtables_dispatch_ = 1
CCDCamera_vtables_ = [
	(( u'Info' , u'pVal' , ), 1, (1, (), [ (16393, 10, None, "IID('{024DED60-B124-4514-BFE2-02C0F5C51DB9}')") , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'AcqParams' , u'pVal' , ), 2, (2, (), [ (16393, 10, None, "IID('{C03DB779-1345-42AB-9304-95B85789163D}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'AcqParams' , u'pVal' , ), 2, (2, (), [ (9, 1, None, "IID('{C03DB779-1345-42AB-9304-95B85789163D}')") , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

CCDCameraInfo_vtables_dispatch_ = 1
CCDCameraInfo_vtables_ = [
	(( u'Name' , u'pVal' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Width' , u'pVal' , ), 2, (2, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Height' , u'pVal' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'PixelSize' , u'pVal' , ), 4, (4, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Binnings' , u'pVal' , ), 5, (5, (), [ (24579, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ShutterModes' , u'pVal' , ), 6, (6, (), [ (24579, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'ShutterMode' , u'pVal' , ), 7, (7, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'ShutterMode' , u'pVal' , ), 7, (7, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'BinningsAsVariant' , u'pVal' , ), 8, (8, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'ShutterModesAsVariant' , u'pVal' , ), 9, (9, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
]

CCDCameras_vtables_dispatch_ = 1
CCDCameras_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pVal' , ), 0, (0, (), [ (12, 1, None, None) , 
			(16393, 10, None, "IID('{E44E1565-4131-4937-B273-78219E090845}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

Camera_vtables_dispatch_ = 1
Camera_vtables_ = [
	(( u'TakeExposure' , ), 5, (5, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Stock' , u'pVal' , ), 10, (10, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'MainScreen' , u'pVal' , ), 11, (11, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'MainScreen' , u'pVal' , ), 11, (11, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'IsSmallScreenDown' , u'pVal' , ), 12, (12, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'MeasuredExposureTime' , u'pET' , ), 13, (13, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'FilmText' , u'pVal' , ), 14, (14, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'FilmText' , u'pVal' , ), 14, (14, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'ManualExposureTime' , u'pVal' , ), 15, (15, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'ManualExposureTime' , u'pVal' , ), 15, (15, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'PlateuMarker' , u'pVal' , ), 17, (17, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'PlateuMarker' , u'pVal' , ), 17, (17, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'ExposureNumber' , u'pVal' , ), 18, (18, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'ExposureNumber' , u'pVal' , ), 18, (18, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Usercode' , u'pVal' , ), 19, (19, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Usercode' , u'pVal' , ), 19, (19, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'ManualExposure' , u'ps' , ), 21, (21, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'ManualExposure' , u'ps' , ), 21, (21, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'PlateLabelDateType' , u'pVal' , ), 22, (22, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'PlateLabelDateType' , u'pVal' , ), 22, (22, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'ScreenDim' , u'pVal' , ), 23, (23, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'ScreenDim' , u'pVal' , ), 23, (23, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'ScreenDimText' , u'pVal' , ), 24, (24, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'ScreenDimText' , u'pVal' , ), 24, (24, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'ScreenCurrent' , u'pSC' , ), 25, (25, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
]

Configuration_vtables_dispatch_ = 1
Configuration_vtables_ = [
	(( u'ProductFamily' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
]

Gauge_vtables_dispatch_ = 1
Gauge_vtables_ = [
	(( u'Read' , ), 1, (1, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Name' , u'pVal' , ), 10, (10, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Pressure' , u'pPresure' , ), 11, (11, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Status' , u'pVal' , ), 12, (12, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'PressureLevel' , u'pVal' , ), 13, (13, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

Gauges_vtables_dispatch_ = 1
Gauges_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pG' , ), 0, (0, (), [ (12, 0, None, None) , 
			(16393, 10, None, "IID('{52020820-18BF-11D3-86E1-00C04FC126DD}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

Gun_vtables_dispatch_ = 1
Gun_vtables_ = [
	(( u'HTState' , u'ps' , ), 10, (10, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'HTState' , u'ps' , ), 10, (10, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'HTValue' , u'phtval' , ), 11, (11, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'HTValue' , u'phtval' , ), 11, (11, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'HTMaxValue' , u'pMaxHT' , ), 12, (12, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Shift' , u'pBS' , ), 13, (13, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Shift' , u'pBS' , ), 13, (13, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Tilt' , u'pDFT' , ), 14, (14, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Tilt' , u'pDFT' , ), 14, (14, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
]

IUserButton_vtables_dispatch_ = 1
IUserButton_vtables_ = [
	(( u'Name' , u'pName' , ), 10, (10, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Label' , u'pName' , ), 11, (11, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Assignment' , u'pas' , ), 12, (12, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Assignment' , u'pas' , ), 12, (12, (), [ (8, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
]

Illumination_vtables_dispatch_ = 1
Illumination_vtables_ = [
	(( u'Normalize' , u'nm' , ), 1, (1, (), [ (3, 0, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pMode' , ), 11, (11, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pMode' , ), 11, (11, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'SpotsizeIndex' , u'pSS' , ), 12, (12, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'SpotsizeIndex' , u'pSS' , ), 12, (12, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Intensity' , u'pInt' , ), 13, (13, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Intensity' , u'pInt' , ), 13, (13, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'IntensityZoomEnabled' , u'pIZE' , ), 14, (14, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'IntensityZoomEnabled' , u'pIZE' , ), 14, (14, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'IntensityLimitEnabled' , u'pILE' , ), 15, (15, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'IntensityLimitEnabled' , u'pILE' , ), 15, (15, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'BeamBlanked' , u'pBB' , ), 16, (16, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'BeamBlanked' , u'pBB' , ), 16, (16, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'Shift' , u'pBS' , ), 17, (17, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'Shift' , u'pBS' , ), 17, (17, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'Tilt' , u'pDFT' , ), 18, (18, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'Tilt' , u'pDFT' , ), 18, (18, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'RotationCenter' , u'pRC' , ), 19, (19, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'RotationCenter' , u'pRC' , ), 19, (19, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'CondenserStigmator' , u'pCStig' , ), 20, (20, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'CondenserStigmator' , u'pCStig' , ), 20, (20, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'DFMode' , u'pVal' , ), 21, (21, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'DFMode' , u'pVal' , ), 21, (21, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'CondenserMode' , u'pConMode' , ), 22, (22, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'CondenserMode' , u'pConMode' , ), 22, (22, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'IlluminatedArea' , u'pIll' , ), 23, (23, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'IlluminatedArea' , u'pIll' , ), 23, (23, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'ProbeDefocus' , u'pDef' , ), 24, (24, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'ConvergenceAngle' , u'pAng' , ), 25, (25, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'StemMagnification' , u'pMag' , ), 26, (26, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'StemMagnification' , u'pMag' , ), 26, (26, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'StemRotation' , u'pVal' , ), 27, (27, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'StemRotation' , u'pVal' , ), 27, (27, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'StemFullScanFieldOfView' , u'fov' , ), 28, (28, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
]

InstrumentInterface_vtables_dispatch_ = 1
InstrumentInterface_vtables_ = [
	(( u'NormalizeAll' , ), 1, (1, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'AutoNormalizeEnabled' , u'pANE' , ), 2, (2, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'AutoNormalizeEnabled' , u'pANE' , ), 2, (2, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'ReturnError' , u'TE' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 40 , (3, 0, None, None) , 64 , )),
	(( u'Vector' , u'pVector' , ), 11, (11, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'StagePosition' , u'pStp' , ), 12, (12, (), [ (16393, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Vacuum' , u'pVac' , ), 20, (20, (), [ (16393, 10, None, "IID('{C7646442-1115-11D3-AE00-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Camera' , u'pCamera' , ), 21, (21, (), [ (16393, 10, None, "IID('{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'Stage' , u'pStage' , ), 22, (22, (), [ (16393, 10, None, "IID('{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'Illumination' , u'pI' , ), 23, (23, (), [ (16393, 10, None, "IID('{EF960690-1C38-11D3-AE0B-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'Projection' , u'pP' , ), 24, (24, (), [ (16393, 10, None, "IID('{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'Gun' , u'pG' , ), 25, (25, (), [ (16393, 10, None, "IID('{E6F00870-3164-11D3-B4C8-00A024CB9221}')") , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'UserButtons' , u'pUBS' , ), 26, (26, (), [ (16393, 10, None, "IID('{50C21D10-317F-11D3-B4C8-00A024CB9221}')") , ], 1 , 2 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'AutoLoader' , u'pAL' , ), 27, (27, (), [ (16393, 10, None, "IID('{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}')") , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'TemperatureControl' , u'pTC' , ), 28, (28, (), [ (16393, 10, None, "IID('{71B6E709-B21F-435F-9529-1AEE55CFA029}')") , ], 1 , 2 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'BlankerShutter' , u'pBS' , ), 29, (29, (), [ (16393, 10, None, "IID('{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}')") , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'InstrumentModeControl' , u'pIMC' , ), 30, (30, (), [ (16393, 10, None, "IID('{8DC0FC71-FF15-40D8-8174-092218D8B76B}')") , ], 1 , 2 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'Acquisition' , u'pIAcq' , ), 31, (31, (), [ (16393, 10, None, "IID('{D6BBF89C-22B8-468F-80A1-947EA89269CE}')") , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'Configuration' , u'pIConfig' , ), 32, (32, (), [ (16393, 10, None, "IID('{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}')") , ], 1 , 2 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'ApertureMechanismCollection' , u'pApertureMechanismCollection' , ), 33, (33, (), [ (16393, 10, None, "IID('{A5015A2D-0436-472E-9140-43DC805B1EEE}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
]

InstrumentModeControl_vtables_dispatch_ = 1
InstrumentModeControl_vtables_ = [
	(( u'StemAvailable' , u'pAvail' , ), 10, (10, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'InstrumentMode' , u'pMode' , ), 11, (11, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'InstrumentMode' , u'pMode' , ), 11, (11, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

Projection_vtables_dispatch_ = 1
Projection_vtables_ = [
	(( u'ResetDefocus' , ), 1, (1, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Normalize' , u'norm' , ), 2, (2, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'ChangeProjectionIndex' , u'addVal' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 1 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pVal' , ), 10, (10, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Mode' , u'pVal' , ), 10, (10, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Focus' , u'pVal' , ), 11, (11, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Focus' , u'pVal' , ), 11, (11, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Magnification' , u'pVal' , ), 12, (12, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'CameraLength' , u'pVal' , ), 13, (13, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'MagnificationIndex' , u'pVal' , ), 14, (14, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'MagnificationIndex' , u'pVal' , ), 14, (14, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'CameraLengthIndex' , u'pVal' , ), 15, (15, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
	(( u'CameraLengthIndex' , u'pVal' , ), 15, (15, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 76 , (3, 0, None, None) , 0 , )),
	(( u'ImageShift' , u'pVal' , ), 16, (16, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 80 , (3, 0, None, None) , 0 , )),
	(( u'ImageShift' , u'pVal' , ), 16, (16, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 84 , (3, 0, None, None) , 0 , )),
	(( u'ImageBeamShift' , u'pVal' , ), 17, (17, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 88 , (3, 0, None, None) , 0 , )),
	(( u'ImageBeamShift' , u'pVal' , ), 17, (17, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 92 , (3, 0, None, None) , 0 , )),
	(( u'DiffractionShift' , u'pVal' , ), 18, (18, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 96 , (3, 0, None, None) , 0 , )),
	(( u'DiffractionShift' , u'pVal' , ), 18, (18, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 100 , (3, 0, None, None) , 0 , )),
	(( u'DiffractionStigmator' , u'pVal' , ), 19, (19, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 104 , (3, 0, None, None) , 0 , )),
	(( u'DiffractionStigmator' , u'pVal' , ), 19, (19, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 108 , (3, 0, None, None) , 0 , )),
	(( u'ObjectiveStigmator' , u'pVal' , ), 20, (20, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 112 , (3, 0, None, None) , 0 , )),
	(( u'ObjectiveStigmator' , u'pVal' , ), 20, (20, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 116 , (3, 0, None, None) , 0 , )),
	(( u'Defocus' , u'pVal' , ), 21, (21, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 120 , (3, 0, None, None) , 0 , )),
	(( u'Defocus' , u'pVal' , ), 21, (21, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 124 , (3, 0, None, None) , 0 , )),
	(( u'SubModeString' , u'pVal' , ), 22, (22, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 128 , (3, 0, None, None) , 0 , )),
	(( u'SubMode' , u'pVal' , ), 23, (23, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 132 , (3, 0, None, None) , 0 , )),
	(( u'SubModeMinIndex' , u'pN' , ), 24, (24, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 136 , (3, 0, None, None) , 0 , )),
	(( u'SubModeMaxIndex' , u'pN' , ), 25, (25, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 140 , (3, 0, None, None) , 0 , )),
	(( u'ObjectiveExcitation' , u'pVal' , ), 26, (26, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 144 , (3, 0, None, None) , 0 , )),
	(( u'ProjectionIndex' , u'pVal' , ), 27, (27, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 148 , (3, 0, None, None) , 0 , )),
	(( u'ProjectionIndex' , u'pVal' , ), 27, (27, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 152 , (3, 0, None, None) , 0 , )),
	(( u'LensProgram' , u'pVal' , ), 28, (28, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 156 , (3, 0, None, None) , 0 , )),
	(( u'LensProgram' , u'pVal' , ), 28, (28, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 160 , (3, 0, None, None) , 0 , )),
	(( u'ImageRotation' , u'pVal' , ), 29, (29, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 164 , (3, 0, None, None) , 0 , )),
	(( u'DetectorShift' , u'pVal' , ), 30, (30, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 168 , (3, 0, None, None) , 0 , )),
	(( u'DetectorShift' , u'pVal' , ), 30, (30, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 172 , (3, 0, None, None) , 0 , )),
	(( u'DetectorShiftMode' , u'pVal' , ), 31, (31, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 176 , (3, 0, None, None) , 0 , )),
	(( u'DetectorShiftMode' , u'pVal' , ), 31, (31, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 180 , (3, 0, None, None) , 0 , )),
	(( u'ImageBeamTilt' , u'pVal' , ), 32, (32, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 184 , (3, 0, None, None) , 0 , )),
	(( u'ImageBeamTilt' , u'pVal' , ), 32, (32, (), [ (9, 1, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 4 , 4 , 0 , 188 , (3, 0, None, None) , 0 , )),
]

STEMAcqParams_vtables_dispatch_ = 1
STEMAcqParams_vtables_ = [
	(( u'ImageSize' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'ImageSize' , u'pVal' , ), 1, (1, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'DwellTime' , u'pVal' , ), 2, (2, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'DwellTime' , u'pVal' , ), 2, (2, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Binning' , u'pVal' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Binning' , u'pVal' , ), 3, (3, (), [ (3, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'MaxResolution' , u'p' , ), 4, (4, (), [ (16393, 10, None, "IID('{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

STEMDetector_vtables_dispatch_ = 1
STEMDetector_vtables_ = [
	(( u'Info' , u'pVal' , ), 1, (1, (), [ (16393, 10, None, "IID('{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}')") , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
]

STEMDetectorInfo_vtables_dispatch_ = 1
STEMDetectorInfo_vtables_ = [
	(( u'Name' , u'pVal' , ), 1, (1, (), [ (16392, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Brightness' , u'pVal' , ), 2, (2, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Brightness' , u'pVal' , ), 2, (2, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Contrast' , u'pVal' , ), 3, (3, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Contrast' , u'pVal' , ), 3, (3, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Binnings' , u'pVal' , ), 4, (4, (), [ (24579, 10, None, None) , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'BinningsAsVariant' , u'pVal' , ), 5, (5, (), [ (16396, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

STEMDetectors_vtables_dispatch_ = 1
STEMDetectors_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pVal' , ), 0, (0, (), [ (12, 1, None, None) , 
			(16393, 10, None, "IID('{D77C0D65-A1DD-4D0A-AF25-C280046A5719}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'AcqParams' , u'pVal' , ), 2, (2, (), [ (16393, 10, None, "IID('{DDC14710-6152-4963-AEA4-C67BA784C6B4}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'AcqParams' , u'pVal' , ), 2, (2, (), [ (9, 1, None, "IID('{DDC14710-6152-4963-AEA4-C67BA784C6B4}')") , ], 1 , 4 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

Stage_vtables_dispatch_ = 1
Stage_vtables_ = [
	(( u'Goto' , u'newPos' , u'mask' , ), 1, (1, (), [ (9, 0, None, "IID('{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}')") , 
			(3, 0, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'MoveTo' , u'newPos' , u'mask' , ), 2, (2, (), [ (9, 0, None, "IID('{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}')") , 
			(3, 0, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Status' , u'pVal' , ), 10, (10, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Position' , u'pVal' , ), 11, (11, (), [ (16393, 10, None, "IID('{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Holder' , u'pVal' , ), 12, (12, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'AxisData' , u'mask' , u'pVal' , ), 13, (13, (), [ (3, 0, None, None) , 
			(16393, 10, None, "IID('{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}')") , ], 1 , 2 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'GotoWithSpeed' , u'newPos' , u'mask' , u'speed' , ), 3, (3, (), [ 
			(9, 1, None, "IID('{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}')") , (3, 1, None, None) , (5, 1, None, None) , ], 1 , 1 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
]

StageAxisData_vtables_dispatch_ = 1
StageAxisData_vtables_ = [
	(( u'MinPos' , u'pVal' , ), 1, (1, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'MaxPos' , u'pVal' , ), 2, (2, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'UnitType' , u'pVal' , ), 3, (3, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

StagePosition_vtables_dispatch_ = 1
StagePosition_vtables_ = [
	(( u'GetAsArray' , u'pos' , ), 1, (1, (), [ (16389, 0, None, None) , ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'SetAsArray' , u'pos' , ), 2, (2, (), [ (16389, 0, None, None) , ], 1 , 1 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'X' , u'pVal' , ), 10, (10, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'X' , u'pVal' , ), 10, (10, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'Y' , u'pVal' , ), 11, (11, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'Y' , u'pVal' , ), 11, (11, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
	(( u'Z' , u'pVal' , ), 12, (12, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 52 , (3, 0, None, None) , 0 , )),
	(( u'Z' , u'pVal' , ), 12, (12, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 56 , (3, 0, None, None) , 0 , )),
	(( u'A' , u'pVal' , ), 13, (13, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 60 , (3, 0, None, None) , 0 , )),
	(( u'A' , u'pVal' , ), 13, (13, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 64 , (3, 0, None, None) , 0 , )),
	(( u'B' , u'pVal' , ), 14, (14, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 68 , (3, 0, None, None) , 0 , )),
	(( u'B' , u'pVal' , ), 14, (14, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 72 , (3, 0, None, None) , 0 , )),
]

TemperatureControl_vtables_dispatch_ = 1
TemperatureControl_vtables_ = [
	(( u'ForceRefill' , ), 1, (1, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'TemperatureControlAvailable' , u'pAvail' , ), 10, (10, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'RefrigerantLevel' , u'rl' , u'pVal' , ), 11, (11, (), [ (3, 1, None, None) , 
			(16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'DewarsRemainingTime' , u'time' , ), 13, (13, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'DewarsAreBusyFilling' , u'pVal' , ), 14, (14, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
]

UserButtons_vtables_dispatch_ = 1
UserButtons_vtables_ = [
	(( u'Count' , u'pVal' , ), 1, (1, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Item' , u'index' , u'pUB' , ), 0, (0, (), [ (12, 0, None, None) , 
			(16393, 10, None, "IID('{E6F00871-3164-11D3-B4C8-00A024CB9221}')") , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'_NewEnum' , u'pVal' , ), -4, (-4, (), [ (16397, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
]

Vacuum_vtables_dispatch_ = 1
Vacuum_vtables_ = [
	(( u'RunBufferCycle' , ), 3, (3, (), [ ], 1 , 1 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'Status' , u'pVal' , ), 10, (10, (), [ (16387, 10, None, None) , ], 1 , 2 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'PVPRunning' , u'pVal' , ), 11, (11, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Gauges' , u'pG' , ), 12, (12, (), [ (16393, 10, None, "IID('{6E6F03B0-2ECE-11D3-AE79-004095005B07}')") , ], 1 , 2 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
	(( u'ColumnValvesOpen' , u'pO' , ), 13, (13, (), [ (16395, 10, None, None) , ], 1 , 2 , 4 , 0 , 44 , (3, 0, None, None) , 0 , )),
	(( u'ColumnValvesOpen' , u'pO' , ), 13, (13, (), [ (11, 1, None, None) , ], 1 , 4 , 4 , 0 , 48 , (3, 0, None, None) , 0 , )),
]

Vector_vtables_dispatch_ = 1
Vector_vtables_ = [
	(( u'X' , u'pVal' , ), 1, (1, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 28 , (3, 0, None, None) , 0 , )),
	(( u'X' , u'pVal' , ), 1, (1, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 32 , (3, 0, None, None) , 0 , )),
	(( u'Y' , u'pVal' , ), 2, (2, (), [ (16389, 10, None, None) , ], 1 , 2 , 4 , 0 , 36 , (3, 0, None, None) , 0 , )),
	(( u'Y' , u'pVal' , ), 2, (2, (), [ (5, 1, None, None) , ], 1 , 4 , 4 , 0 , 40 , (3, 0, None, None) , 0 , )),
]

RecordMap = {
}

CLSIDToClassMap = {
	'{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}' : Configuration,
	'{DD08F876-9FBB-4235-A47E-FF26ADA00900}' : ApertureCollection,
	'{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}' : STEMDetectorInfo,
	'{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}' : Stage,
	'{02CDC9A1-1F1D-11D3-AE11-00A024CBA50C}' : Instrument,
	'{02CDC9A2-1F1D-11D3-AE11-00A024CBA50C}' : UserButtonEvent,
	'{71B6E709-B21F-435F-9529-1AEE55CFA029}' : TemperatureControl,
	'{EF960690-1C38-11D3-AE0B-00A024CBA50C}' : Illumination,
	'{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}' : BlankerShutter,
	'{8DC0FC71-FF15-40D8-8174-092218D8B76B}' : InstrumentModeControl,
	'{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}' : AutoLoader,
	'{86C13CE3-934E-47DE-A211-2009E10E1EE1}' : ApertureMechanism,
	'{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}' : StageAxisData,
	'{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}' : Vector,
	'{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}' : StagePosition,
	'{C7646442-1115-11D3-AE00-00A024CBA50C}' : Vacuum,
	'{E6F00870-3164-11D3-B4C8-00A024CB9221}' : Gun,
	'{E6F00871-3164-11D3-B4C8-00A024CB9221}' : IUserButton,
	'{52020820-18BF-11D3-86E1-00C04FC126DD}' : Gauge,
	'{86365241-4D38-4642-B024-CF450CEB250B}' : AcqImages,
	'{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}' : Aperture,
	'{DDC14710-6152-4963-AEA4-C67BA784C6B4}' : STEMAcqParams,
	'{024DED60-B124-4514-BFE2-02C0F5C51DB9}' : CCDCameraInfo,
	'{A5015A2D-0436-472E-9140-43DC805B1EEE}' : ApertureMechanismCollection,
	'{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}' : Camera,
	'{6E6F03B0-2ECE-11D3-AE79-004095005B07}' : Gauges,
	'{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}' : CCDCameras,
	'{3A4CE1F0-3A05-11D3-AE81-004095005B07}' : UserButton,
	'{BC0A2B11-10FF-11D3-AE00-00A024CBA50C}' : InstrumentInterface,
	'{35A2675D-E67B-4834-8940-85E7833C61A6}' : STEMDetectors,
	'{E44E1565-4131-4937-B273-78219E090845}' : CCDCamera,
	'{50C21D10-317F-11D3-B4C8-00A024CB9221}' : UserButtons,
	'{C03DB779-1345-42AB-9304-95B85789163D}' : CCDAcqParams,
	'{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}' : Projection,
	'{D77C0D65-A1DD-4D0A-AF25-C280046A5719}' : STEMDetector,
	'{D6BBF89C-22B8-468F-80A1-947EA89269CE}' : Acquisition,
	'{E15F4810-43C6-489A-9E8A-588B0949E153}' : AcqImage,
}
CLSIDToPackageMap = {}
win32com.client.CLSIDToClass.RegisterCLSIDsFromDict( CLSIDToClassMap )
VTablesToPackageMap = {}
VTablesToClassMap = {
	'{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}' : 'Configuration',
	'{DD08F876-9FBB-4235-A47E-FF26ADA00900}' : 'ApertureCollection',
	'{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}' : 'STEMDetectorInfo',
	'{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}' : 'Stage',
	'{71B6E709-B21F-435F-9529-1AEE55CFA029}' : 'TemperatureControl',
	'{EF960690-1C38-11D3-AE0B-00A024CBA50C}' : 'Illumination',
	'{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}' : 'BlankerShutter',
	'{8DC0FC71-FF15-40D8-8174-092218D8B76B}' : 'InstrumentModeControl',
	'{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}' : 'AutoLoader',
	'{86C13CE3-934E-47DE-A211-2009E10E1EE1}' : 'ApertureMechanism',
	'{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}' : 'StageAxisData',
	'{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}' : 'Vector',
	'{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}' : 'StagePosition',
	'{C7646442-1115-11D3-AE00-00A024CBA50C}' : 'Vacuum',
	'{E6F00870-3164-11D3-B4C8-00A024CB9221}' : 'Gun',
	'{E6F00871-3164-11D3-B4C8-00A024CB9221}' : 'IUserButton',
	'{52020820-18BF-11D3-86E1-00C04FC126DD}' : 'Gauge',
	'{86365241-4D38-4642-B024-CF450CEB250B}' : 'AcqImages',
	'{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}' : 'Aperture',
	'{DDC14710-6152-4963-AEA4-C67BA784C6B4}' : 'STEMAcqParams',
	'{024DED60-B124-4514-BFE2-02C0F5C51DB9}' : 'CCDCameraInfo',
	'{A5015A2D-0436-472E-9140-43DC805B1EEE}' : 'ApertureMechanismCollection',
	'{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}' : 'Camera',
	'{6E6F03B0-2ECE-11D3-AE79-004095005B07}' : 'Gauges',
	'{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}' : 'CCDCameras',
	'{BC0A2B11-10FF-11D3-AE00-00A024CBA50C}' : 'InstrumentInterface',
	'{35A2675D-E67B-4834-8940-85E7833C61A6}' : 'STEMDetectors',
	'{E44E1565-4131-4937-B273-78219E090845}' : 'CCDCamera',
	'{50C21D10-317F-11D3-B4C8-00A024CB9221}' : 'UserButtons',
	'{C03DB779-1345-42AB-9304-95B85789163D}' : 'CCDAcqParams',
	'{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}' : 'Projection',
	'{D77C0D65-A1DD-4D0A-AF25-C280046A5719}' : 'STEMDetector',
	'{D6BBF89C-22B8-468F-80A1-947EA89269CE}' : 'Acquisition',
	'{E15F4810-43C6-489A-9E8A-588B0949E153}' : 'AcqImage',
}


NamesToIIDMap = {
	'InstrumentModeControl' : '{8DC0FC71-FF15-40D8-8174-092218D8B76B}',
	'CCDCamera' : '{E44E1565-4131-4937-B273-78219E090845}',
	'UserButtonEvent' : '{02CDC9A2-1F1D-11D3-AE11-00A024CBA50C}',
	'IUserButton' : '{E6F00871-3164-11D3-B4C8-00A024CB9221}',
	'ApertureMechanismCollection' : '{A5015A2D-0436-472E-9140-43DC805B1EEE}',
	'AcqImages' : '{86365241-4D38-4642-B024-CF450CEB250B}',
	'Camera' : '{9851BC41-1B8C-11D3-AE0A-00A024CBA50C}',
	'CCDCameras' : '{C851D96C-96B2-4BDF-8DF2-C0A01B76E265}',
	'STEMDetectorInfo' : '{96DE094B-9CDC-4796-8697-E7DD5DC3EC3F}',
	'STEMAcqParams' : '{DDC14710-6152-4963-AEA4-C67BA784C6B4}',
	'Aperture' : '{CBF4E5B8-378D-43DD-9C58-F588D5E3444B}',
	'StageAxisData' : '{8F1E91C2-B97D-45B8-87C9-423F5EB10B8A}',
	'ApertureCollection' : '{DD08F876-9FBB-4235-A47E-FF26ADA00900}',
	'TemperatureControl' : '{71B6E709-B21F-435F-9529-1AEE55CFA029}',
	'Vacuum' : '{C7646442-1115-11D3-AE00-00A024CBA50C}',
	'Configuration' : '{39CACDAF-F47C-4BBF-9FFA-A7A737664CED}',
	'Stage' : '{E7AE1E41-1BF8-11D3-AE0B-00A024CBA50C}',
	'CCDCameraInfo' : '{024DED60-B124-4514-BFE2-02C0F5C51DB9}',
	'STEMDetectors' : '{35A2675D-E67B-4834-8940-85E7833C61A6}',
	'AutoLoader' : '{28DF27EA-2058-41D0-ABBD-167FB3BFCD8F}',
	'InstrumentInterface' : '{BC0A2B11-10FF-11D3-AE00-00A024CBA50C}',
	'Gun' : '{E6F00870-3164-11D3-B4C8-00A024CB9221}',
	'CCDAcqParams' : '{C03DB779-1345-42AB-9304-95B85789163D}',
	'Vector' : '{9851BC47-1B8C-11D3-AE0A-00A024CBA50C}',
	'BlankerShutter' : '{F1F59BB0-F8A0-439D-A3BF-87F527B600C4}',
	'Acquisition' : '{D6BBF89C-22B8-468F-80A1-947EA89269CE}',
	'STEMDetector' : '{D77C0D65-A1DD-4D0A-AF25-C280046A5719}',
	'Projection' : '{B39C3AE1-1E41-11D3-AE0E-00A024CBA50C}',
	'Gauges' : '{6E6F03B0-2ECE-11D3-AE79-004095005B07}',
	'AcqImage' : '{E15F4810-43C6-489A-9E8A-588B0949E153}',
	'UserButtons' : '{50C21D10-317F-11D3-B4C8-00A024CB9221}',
	'Gauge' : '{52020820-18BF-11D3-86E1-00C04FC126DD}',
	'Illumination' : '{EF960690-1C38-11D3-AE0B-00A024CBA50C}',
	'ApertureMechanism' : '{86C13CE3-934E-47DE-A211-2009E10E1EE1}',
	'StagePosition' : '{9851BC4A-1B8C-11D3-AE0A-00A024CBA50C}',
}

win32com.client.constants.__dicts__.append(constants.__dict__)

