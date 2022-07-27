    #include "volFieldsFwd.H"

    volScalarField Phi
    (
        IOobject
        (
            "Phi",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );


    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );

    volVectorField U_nn("U_nn", U);
//    volVectorField U_nn
//    (
//        IOobject
//        (
//            "U_nn",
//            runTime.timeName(),
//            mesh,
//            IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        mesh,
//        dimensionedVector("U_nn", U.dimensions(), vector(1,0,0))
//      );


    volScalarField Phi_nn("Phi_nn", Phi);
//  volScalarField Phi_nn
//    (
//        IOobject
//        (
//            "Phi_nn",
//            runTime.timeName(),
//            mesh,
//            IOobject::NO_READ,
//            IOobject::AUTO_WRITE
//        ),
//        mesh,
//
//        dimensionedScalar("Phi_nn", Phi.dimensions(), 0)
//      );

    volScalarField error_Phi
    (
        IOobject
        (
            "error_Phi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("error_Phi", Phi.dimensions(), 0)
    );


    volScalarField error_U
    (
        IOobject
        (
            "error_U",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("error_U", U.dimensions(), 0)
    );
