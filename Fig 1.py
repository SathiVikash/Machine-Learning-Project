from gaphor import UML
from gaphor.core.modeling import ElementFactory, Diagram, StyleSheet
from gaphor.diagram.drop import drop
from gaphor.extensions.ipython import auto_layout
from gaphor.storage import storage


# --------------------------------------------------
# BASIC FUNCTIONS
# --------------------------------------------------

element_factory = ElementFactory()

element_factory.create(StyleSheet)


def create_class(name, attributes=[], methods=[]):
    c = element_factory.create(UML.Class)
    c.name = name

    for attr_name, attr_type in attributes:
        attr = element_factory.create(UML.Property)
        attr.name = attr_name
        attr.typeValue = attr_type
        c.ownedAttribute = attr

    for method_name, return_type in methods:
        op = element_factory.create(UML.Operation)
        op.name = method_name
        op.returnResult = return_type
        c.ownedOperation = op

    return c


def generalization(parent, child):
    g = element_factory.create(UML.Generalization)
    g.general = parent
    g.specific = child
    return g


def dependency(client, supplier):
    d = element_factory.create(UML.Dependency)
    d.client = client
    d.supplier = supplier
    return d


def association(class1, class2):
    a = element_factory.create(UML.Association)

    end1 = element_factory.create(UML.Property)
    end1.type = class1

    end2 = element_factory.create(UML.Property)
    end2.type = class2

    a.memberEnd = end1
    a.memberEnd = end2

    return a


# ==================================================
# ADAPTER PATTERN
# ==================================================

smart_device = create_class(
    "SmartDevice",
    [
        ("id", "string"),
        ("name", "string"),
        ("type", "DeviceType"),
        ("roomId", "string"),
        ("roomName", "string"),
        ("powerWatts", "number")
    ],
    [
        ("turnOn()", "void"),
        ("turnOff()", "void"),
        ("getState()", "Record<string, any>"),
        ("setPowerWatts(number)", "void")
    ]
)

smart_device.isAbstract = True


external_adapter = create_class(
    "ExternalDeviceAdapter",
    [
        ("adaptee", "ExternalVendorDevice"),
        ("protocolTranslation", "string"),
        ("lastParsedTelemetry", "VendorTelemetry | null")
    ],
    [
        ("turnOn()", "void"),
        ("turnOff()", "void"),
        ("getState()", "Record<string, any>"),
        ("getTranslationLog()", "string[]"),
        ("getAdapteeMetadata()", "VendorMetadata")
    ]
)


external_vendor = create_class(
    "ExternalVendorDevice",
    [
        ("serialNumber", "string"),
        ("protocolVersion", "string"),
        ("registers", "Map<string, number>")
    ],
    [
        ("sendHexCommand(opcode, num, val)", "boolean"),
        ("readModbus(register)", "number"),
        ("writeModbus(register, val)", "void"),
        ("parseRawPayload(raw)", "VendorTelemetry"),
        ("getHardwareSerial()", "string")
    ]
)


adapter_client = create_class(
    "Client",
    [],
    [
        ("invoke()", "void")
    ]
)


adapter_generalization = generalization(
    smart_device,
    external_adapter
)

adapter_dependency = dependency(
    adapter_client,
    smart_device
)

adapter_association = association(
    external_adapter,
    external_vendor
)


adapter_diagram = element_factory.create(Diagram)
adapter_diagram.name = "Adapter Pattern - Smart Home"


for element in [
    smart_device,
    external_adapter,
    external_vendor,
    adapter_client,
    adapter_generalization,
    adapter_dependency,
    adapter_association
]:
    drop(element, adapter_diagram, x=0, y=0)


auto_layout(adapter_diagram)


# ==================================================
# FACADE PATTERN
# ==================================================

facade = create_class(
    "SmartHomeFacade",
    [
        ("deviceManager", "DeviceManager"),
        ("automationEngine", "AutomationEngine"),
        ("contextSubject", "ContextSubject"),
        ("energyMonitor", "EnergyMonitor")
    ],
    [
        ("orchestrateMorningRoutine()", "FacadeExecutionResult"),
        ("orchestrateAllAwaySecurityLockdown()", "FacadeExecutionResult"),
        ("orchestrateErrorMode()", "FacadeExecutionResult"),
        ("handleSensorTelemetry(sensorId, state, value)", "void"),
        ("getSystemHealthSummary()", "SystemHealthReport"),
        ("executeManualOverride(deviceId, command)", "void")
    ]
)


device_manager = create_class(
    "DeviceManager",
    [
        ("devices", "Map<string, SmartDevice>"),
        ("roomIndex", "Map<string, List<string>>")
    ],
    [
        ("getDevice(id)", "SmartDevice"),
        ("registerDevice(device)", "void"),
        ("getDevicesByRoom(roomId)", "List<SmartDevice>"),
        ("setDeviceState(id, state)", "void"),
        ("turnOffAllDevices()", "void")
    ]
)


context_subject = create_class(
    "ContextSubject",
    [
        ("observers", "List<ContextObserver>")
    ],
    [
        ("attach(obs)", "void"),
        ("detach(obs)", "void"),
        ("notifyObservers()", "void"),
        ("broadcastAlert(level, msg)", "void"),
        ("getHistory()", "ContextEvent[]")
    ]
)


automation_engine = create_class(
    "AutomationEngine",
    [
        ("activeStrategy", "AutomationStrategy"),
        ("rules", "AutomationRule[]")
    ],
    [
        ("setStrategy(strategy)", "void"),
        ("evaluateRule(context)", "Decision"),
        ("enableRule(id)", "void"),
        ("disableRule(id)", "void")
    ]
)


neighborhood = create_class(
    "Neighborhood",
    [
        ("currentLoadWatts", "number"),
        ("tariff", "number")
    ],
    [
        ("calculateTotalKwh()", "number"),
        ("isPeakPricing()", "boolean"),
        ("getForecast()", "Forecast")
    ]
)


facade_dependency_1 = dependency(
    facade,
    device_manager
)

facade_dependency_2 = dependency(
    facade,
    automation_engine
)

facade_dependency_3 = dependency(
    facade,
    context_subject
)

facade_dependency_4 = dependency(
    facade,
    neighborhood
)


facade_diagram = element_factory.create(Diagram)
facade_diagram.name = "Facade Pattern - Smart Home"


for element in [
    facade,
    device_manager,
    context_subject,
    automation_engine,
    neighborhood,
    facade_dependency_1,
    facade_dependency_2,
    facade_dependency_3,
    facade_dependency_4
]:
    drop(element, facade_diagram, x=0, y=0)


auto_layout(facade_diagram)


# ==================================================
# SAVE
# ==================================================

filename = "SmartHome_Adapter_Facade.gaphor"

with open(filename, "w", encoding="utf-8") as file:
    storage.save(file, element_factory)

print("====================================")
print("Smart Home diagrams created!")
print("====================================")
print("Adapter Pattern - Smart Home")
print("Facade Pattern - Smart Home")
print("Saved as:", filename)