abstract type AbstractMaterial end
abstract type AbstractMaterialCache end
struct NoMaterialCache <: AbstractMaterialCache end

abstract type AbstractGrid end

abstract type AbstractMaterialGroup end

abstract type AbstractBoundaryCondition end
struct NoBoundaryCondition <: AbstractBoundaryCondition end

abstract type AbstractExternalForce end
struct NoExternalForce <: AbstractExternalForce end

abstract type AbstractShapeFunction end
