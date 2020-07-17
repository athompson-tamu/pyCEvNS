// Photon flux analyzer.

#include <fstream>
#include <iostream>

{

  TFile *infile = TFile::Open("data/angular_flux.root");
  TH2D *h1 = (TH2D*)infile->Get("htemp");

  ofstream flux;
  flux.open("data/geant4_flux_DUNE.txt", std::ios_base::app);

  double energy = 0.0;
  double cosine = 0.0;
  double weight = 0.0;
  for (int i=0; i<100; ++i) {
    for (int j=0; j<100; ++j) {
    weight = h1->GetBinContent(i,j);
    energy = ((TAxis*)h1->GetXaxis())->GetBinCenter(i);
    cosine = ((TAxis*)h1->GetYaxis())->GetBinCenter(j);
    printf("E = %f, Cos = %f , wgt = %f \n", energy, cosine, weight);
    flux << energy << "," << cosine << "," << weight << std::endl;

    }
  }


  flux.close();


}
