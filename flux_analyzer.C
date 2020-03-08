// Photon flux analyzer.
{
  #include <fstream>
  #include <iostream>
  using namespace std;

  TFile *infile = TFile::Open("data/3MeVandAbove.root");

  TTree *t = (TTree*)infile->Get("gammaInfo");

  double ke = 0;
  TString *parent = new TString();
  TString *cProcess = new TString();
  Double_t energy;
  Double_t cosine;

  //t->SetBranchAddress("CreationEnergy", &ke);
  //t->SetBranchAddress("CreationProcess", &cProcess);
  t->SetBranchAddress("Parent", &parent);
  t->SetBranchAddress("Energy", &energy);
  t->SetBranchAddress("CosTheta", &cosine);


  t->Print();

  Long64_t n = t->GetEntries();


  ofstream neutron_flux;
  ofstream proton_flux;
  ofstream ep_flux;
  ofstream em_flux;
  ofstream pi0_flux;
  ofstream pim_flux;
  ofstream pip_flux;
  ofstream triton_flux;
  neutron_flux.open("data/neutron_flux.txt", std::ios_base::app);
  proton_flux.open("data/proton_flux.txt", std::ios_base::app);
  ep_flux.open("data/ep_flux.txt", std::ios_base::app);
  em_flux.open("data/em_flux.txt", std::ios_base::app);
  pi0_flux.open("data/pi0_flux.txt", std::ios_base::app);
  pim_flux.open("data/pim_flux.txt", std::ios_base::app);
  pip_flux.open("data/pip_flux.txt", std::ios_base::app);
  triton_flux.open("data/triton_flux.txt", std::ios_base::app);
  for (Long64_t i=0; i<n; i++) {
    t->GetEntry(i);
    //cout << parent->Data() << endl;
    //cout << energy << endl;

    // Fill files
    if (strncmp(parent->Data(),"neutron",2) == 0) neutron_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"proton",2) == 0) proton_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"e+",2) == 0) ep_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"e-",2) == 0) em_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"pi0",3) == 0) pi0_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"pi-",3) == 0) pim_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"pi+",3) == 0) pip_flux << energy << "," << cosine << endl;
    if (strncmp(parent->Data(),"triton",2) == 0) triton_flux << energy << "," << cosine << endl;


  }

  neutron_flux.close();
  proton_flux.close();
  ep_flux.close();
  em_flux.close();
  pi0_flux.close();
  pim_flux.close();
  pip_flux.close();
  triton_flux.close();


}