    #include "volFieldsFwd.H"
    volScalarField vf 
    (
        IOobject
        (
            "vf",
            runTime.timeName(), 
            mesh, 
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volScalarField vf_nn("vf_nn", vf);

    volScalarField error_c 
    (
        IOobject
        (
            "error_c", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedScalar("error_c", vf.dimensions(), 0)
    );
    
    volVectorField gradvf_nn
    (
        IOobject
        (
            "gradvf_nn", 
            runTime.timeName(), 
            mesh, 
            IOobject::NO_READ, 
            IOobject::AUTO_WRITE
        ),
        mesh, 
        dimensionedVector("gradvf_nn", vf.dimensions(), vector(0,0,0))
    );
