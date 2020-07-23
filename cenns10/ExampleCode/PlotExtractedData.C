#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>

#include "TCanvas.h"
#include "TLegend.h"
#include "TColor.h"
#include "TDatime.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TLine.h"
#include "TMultiGraph.h"
#include "TH1.h"
#include "TBox.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TMath.h"

int readFile(std::vector<double> &centerx, std::vector<double> &centery, std::vector<double> &centerz, std::vector<double> &events, std::string filename)
{
  std::string line = "";
  double xbins = 0;
  double ybins = 0;
  double zbins = 0;
  double evs = 0;

  std::ifstream ins(filename.c_str());

  if(!ins.good())
  {
    cout << "Failure!" << endl;
    return -1;
  }
  while(std::getline(ins,line))
  {
    std::stringstream ss(line);
    if(ins.eof()) {break;}   
    ss >> xbins >> ybins >> zbins >> evs;
    centerx.push_back(xbins);
    centery.push_back(ybins);
    centerz.push_back(zbins);
    events.push_back(evs);
    line = "";
    xbins=ybins=zbins=evs=0.0;
  }
  return 0;
}

int readsystFile1d(std::vector<double> &centerx, std::vector<double> &minsyst, std::vector<double> &maxsyst, std::string filename)
{
  std::string line = "";
  double xbins = 0;
  double min = 0;
  double max = 0;

  std::ifstream ins(filename.c_str());

  if(!ins.good())
  {
    cout << "Failure!" << endl;
    return -1;
  }
  while(std::getline(ins,line))
  {
    std::stringstream ss(line);
    if(ins.eof()) {break;}
    ss >> xbins >> min >> max;
    centerx.push_back(xbins);
    minsyst.push_back(min);
    maxsyst.push_back(max);
    line = "";
    xbins=min=max=0.0;
  }
  return 0;
}

int readsystFile3d(std::vector<double> &centerx, std::vector<double> &centery, std::vector<double> &centerz, std::vector<double> &minsyst, std::vector<double> &maxsyst, std::string filename)
{
  std::string line = "";
  double xbins = 0;
  double ybins = 0;
  double zbins = 0;
  double min = 0;
  double max = 0;

  std::ifstream ins(filename.c_str());

  if(!ins.good())
  {
    cout << "Failure!" << endl;
    return -1;
  }
  while(std::getline(ins,line))
  {
    std::stringstream ss(line);
    if(ins.eof()) {break;}
    ss >> xbins >> ybins >> zbins >> min >> max;
    centerx.push_back(xbins);
    centery.push_back(ybins);
    centerz.push_back(zbins);
    minsyst.push_back(min);
    maxsyst.push_back(max);
    line = "";
    xbins=ybins=zbins=min=max=0.0;
  }
  return 0;
}

void PlotExtractedData()
{
  TColor *color1 = gROOT->GetColor(10);
  color1->SetRGB(0.10588,0.61960,0.46666);
  TColor *color2 = gROOT->GetColor(11);
  color2->SetRGB(0.85098,0.38431,0.00784);
  TColor *color3 = gROOT->GetColor(12);
  color3->SetRGB(0.45882,0.43921,0.70196);
  TColor *color4 = gROOT->GetColor(13);
  color4->SetRGB(0.90588,0.16078,0.54117);
  TColor *color5 = gROOT->GetColor(14);
  color5->SetRGB(0.40,0.65098,0.11764);
  
  std::vector<double> centerx;
  std::vector<double> centery;
  std::vector<double> centerz;
  std::vector<double> scrapbins;
  std::vector<double> evsdata;
  std::vector<double> evsbkg;
  std::vector<double> evsbrn;
  std::vector<double> evscevns;
  std::vector<double> evsdelbrn;
  std::vector<double> evsdatabkgsub;
  std::vector<double> energybins;
  std::vector<double> psdbins;
  std::vector<double> timebins;
  std::vector<double> systerrormin1denergy;
  std::vector<double> systerrormax1denergy;
  std::vector<double> systerrormin1dpsd;
  std::vector<double> systerrormax1dpsd;
  std::vector<double> systerrormin1dtime;
  std::vector<double> systerrormax1dtime; 
  readFile(centerx,centery,centerz,evsdata, "../Data/datanobkgsub.txt");
  readFile(scrapbins,scrapbins,scrapbins,evsbkg, "../Data/bkgpdf.txt");
  readFile(scrapbins,scrapbins,scrapbins,evsbrn, "../Data/brnpdf.txt");
  readFile(scrapbins,scrapbins,scrapbins,evsdelbrn, "../Data/delbrnpdf.txt");
  readFile(scrapbins,scrapbins,scrapbins,evscevns, "../Data/cevnspdf.txt");
  readsystFile1d(energybins, systerrormin1denergy, systerrormax1denergy,"../Data/systerrors1denergy.txt");
  readsystFile1d(psdbins, systerrormin1dpsd, systerrormax1dpsd, "../Data/systerrors1dpsd.txt");
  readsystFile1d(timebins, systerrormin1dtime, systerrormax1dtime, "../Data/systerrors1dtime.txt");
 

  double emin = 0;
  double emax = 120;
  double ebins = 12;
  double f90min = 0.5;
  double f90max = 0.9;
  double f90bins = 8;
  double tmin = -0.1;
  double tmax = 4.9;
  double tbins = 10;

  TH3F *data = new TH3F("data", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  TH3F *ssbkg = new TH3F("ssbkg", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  TH3F *brn = new TH3F("brn", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  TH3F *delbrn = new TH3F("delbrn", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  TH3F *cevns = new TH3F("cevns", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  TH3F *ssbkgsubdata = new TH3F("ssbkgsubdata", "", ebins, emin, emax, f90bins, f90min, f90max, tbins, tmin, tmax);
  double cvss = 3152.0;
  double bestfitss = 3131.0;
  for(int i = 0; i < evsdata.size(); ++i)
  {
    data->Fill(centerx[i], centery[i], centerz[i], evsdata[i]);
    ssbkg->Fill(centerx[i], centery[i], centerz[i], evsbkg[i]);
    brn->Fill(centerx[i], centery[i], centerz[i], evsbrn[i]);
    delbrn->Fill(centerx[i], centery[i], centerz[i], evsdelbrn[i]);
    cevns->Fill(centerx[i], centery[i], centerz[i], evscevns[i]);
    double bkgsubdata = evsdata[i] - evsbkg[i]*(bestfitss/cvss);
    ssbkgsubdata->Fill(centerx[i], centery[i], centerz[i], bkgsubdata);
  }

  cout << "Initial CV Predictions: On-beam data: " << data->Integral() << " Steady-State background: " << ssbkg->Integral() << " Prompt BRN: " << brn->Integral() << " Delayed BRN: " << delbrn->Integral() << " CEvNS: " << cevns->Integral() << endl;

  double bestfit_cevns = 159.0;
  double bestfit_brn = 553.0;
  double bestfit_delbrn = 10.0;
  double bestfit_bkg = 3131.0;
  cevns->Scale(1.0/cevns->Integral());
  cevns->Scale(bestfit_cevns);
  brn->Scale(1.0/brn->Integral());
  brn->Scale(bestfit_brn);
  delbrn->Scale(1.0/delbrn->Integral());
  delbrn->Scale(bestfit_delbrn);
  ssbkg->Scale(1.0/ssbkg->Integral());
  ssbkg->Scale(bestfit_bkg);
  TH3F *brnpluscevns = (TH3F*)brn->Clone("brnpluscevns");
  brnpluscevns->Add(cevns);
  brnpluscevns->Add(delbrn);
  cout << "Best-fit Results: On-beam data: " << data->Integral() << " Steady-State background: "<< ssbkg->Integral() << " Prompt BRN: " << brn->Integral() << " Delayed BRN: " << delbrn->Integral() << " CEvNS: " << cevns->Integral() << " BRN+CEvNS: " << brnpluscevns->Integral() << " " << endl;
  TH1F *data_energy = (TH1F*)data->Project3D("x");
  TH1F *data_f90 = (TH1F*)data->Project3D("y");
  TH1F *data_time = (TH1F*)data->Project3D("z");
  TH1F *datasub_energy = (TH1F*)ssbkgsubdata->Project3D("x");
  TH1F *datasub_f90 = (TH1F*)ssbkgsubdata->Project3D("y");
  TH1F *datasub_time = (TH1F*)ssbkgsubdata->Project3D("z");
  
  TH1F *cevns_energy = (TH1F*)cevns->Project3D("x");
  TH1F *cevns_f90 = (TH1F*)cevns->Project3D("y");
  TH1F *cevns_time = (TH1F*)cevns->Project3D("z");
  TH1F *brn_energy = (TH1F*)brn->Project3D("x");
  TH1F *brn_f90 = (TH1F*)brn->Project3D("y");
  TH1F *brn_time = (TH1F*)brn->Project3D("z");
  TH1F *bkg_energy = (TH1F*)ssbkg->Project3D("x");
  TH1F *bkg_f90 = (TH1F*)ssbkg->Project3D("y");
  TH1F *bkg_time = (TH1F*)ssbkg->Project3D("z");
  TH1F *brnpluscevns_energy = (TH1F*)brnpluscevns->Project3D("x");
  TH1F *brnpluscevns_f90 = (TH1F*)brnpluscevns->Project3D("y");
  TH1F *brnpluscevns_time = (TH1F*)brnpluscevns->Project3D("z");

  for(int i = 1; i < datasub_energy->GetNbinsX()+1; ++i)
  {
    double dataval = data_energy->GetBinContent(i);
    double strobeval = bkg_energy->GetBinContent(i);
    double strobeerr = sqrt(strobeval*5.0)/5.0;
    double error = sqrt(dataval + strobeerr*strobeerr);
    datasub_energy->SetBinError(i, error);
  }

  for(int i = 1; i < datasub_f90->GetNbinsX()+1; ++i)
  {
    double dataval = data_f90->GetBinContent(i);
    double strobeval = bkg_f90->GetBinContent(i);
    double strobeerr = sqrt(strobeval*5.0)/5.0;
    double error = sqrt(dataval + strobeerr*strobeerr);
    datasub_f90->SetBinError(i, error);
  }
  
  for(int i = 1; i < datasub_time->GetNbinsX()+1; ++i)
  {
    double dataval = data_time->GetBinContent(i);
    double strobeval = bkg_time->GetBinContent(i);
    double strobeerr = sqrt(strobeval*5.0)/5.0;
    double error = sqrt(dataval + strobeerr*strobeerr);
    datasub_time->SetBinError(i, error);
  }

  std::vector<TBox*> energyerr;
  std::vector<TBox*> psderr;
  std::vector<TBox*> timeerr;

  for(int i = 1; i <= brnpluscevns_time->GetNbinsX(); ++i)
  {
    //cout << "Here" << endl;
    double val = brnpluscevns_time->GetBinContent(i);
    TBox *box = new TBox(brnpluscevns_time->GetBinLowEdge(i), systerrormin1dtime[i-1]*val, brnpluscevns_time->GetBinLowEdge(i+1), systerrormax1dtime[i-1]*val);
    box->SetFillColorAlpha(14, 0.5);
    timeerr.push_back(box);
  }
  for(int i = 1; i <= brnpluscevns_energy->GetNbinsX(); ++i)
  {
    double val = brnpluscevns_energy->GetBinContent(i);
    TBox *box = new TBox(brnpluscevns_energy->GetBinLowEdge(i), systerrormin1denergy[i-1]*val, brnpluscevns_energy->GetBinLowEdge(i+1), systerrormax1denergy[i-1]*val);
    box->SetFillColorAlpha(14,0.5);
    energyerr.push_back(box);
  }
  for(int i = 1; i <= brnpluscevns_f90->GetNbinsX(); ++i)
  {
    double val = brnpluscevns_f90->GetBinContent(i);
    TBox *box = new TBox(brnpluscevns_f90->GetBinLowEdge(i), systerrormin1dpsd[i-1]*val, brnpluscevns_f90->GetBinLowEdge(i+1), systerrormax1dpsd[i-1]*val);
    box->SetFillColorAlpha(14,0.5);
    psderr.push_back(box);
  }
  
  TCanvas *c1 = new TCanvas();
  datasub_energy->SetStats(0);
  cevns_energy->SetStats(0);
  brn_energy->SetStats(0);
  brnpluscevns_energy->SetStats(0);
  datasub_energy->SetMarkerColor(kBlack);
  datasub_energy->SetMarkerStyle(20);
  datasub_energy->SetLineColor(kBlack);
  datasub_energy->SetMarkerSize(1.5);
  cevns_energy->SetLineWidth(2);
  cevns_energy->SetLineColor(12);
  brn_energy->SetLineWidth(2);
  brn_energy->SetLineColor(11);
  brnpluscevns_energy->SetLineWidth(2);
  brnpluscevns_energy->SetLineColor(10);
  datasub_energy->GetXaxis()->SetTitle("Reconstructed Energy (keVee)");
  datasub_energy->GetXaxis()->CenterTitle();
  datasub_energy->SetTitle("");
  datasub_energy->GetXaxis()->SetTitleSize(0.045);
  datasub_energy->GetXaxis()->SetLabelSize(0.040);
  datasub_energy->GetXaxis()->SetTitleOffset(0.9);
  datasub_energy->GetYaxis()->SetLabelSize(0.045);
  double maxaxis = std::max(datasub_energy->GetMaximum(), brnpluscevns_energy->GetMaximum());
  datasub_energy->GetYaxis()->SetRangeUser(-10.0, maxaxis*1.15);
  datasub_energy->Draw("p e0 x0");
  for(int i = 0; i < energyerr.size(); ++i)
    energyerr[i]->Draw("same");
  cevns_energy->Draw("hist same");
  brn_energy->Draw("hist same");
  brnpluscevns_energy->Draw("hist same");
  
  c1->Print("bkgsubenergyExtractedData.pdf");

  TCanvas *c2 = new TCanvas();
  datasub_f90->SetStats(0);
  cevns_f90->SetStats(0);
  brn_f90->SetStats(0);
  brnpluscevns_f90->SetStats(0);
  datasub_f90->SetMarkerColor(kBlack);
  datasub_f90->SetMarkerStyle(20);
  datasub_f90->SetLineColor(kBlack);
  datasub_f90->SetMarkerSize(1.5);
  cevns_f90->SetLineWidth(2);
  cevns_f90->SetLineColor(12);
  brn_f90->SetLineWidth(2);
  brn_f90->SetLineColor(11);
  brnpluscevns_f90->SetLineWidth(2);
  brnpluscevns_f90->SetLineColor(10);
  datasub_f90->SetTitle("");
  datasub_f90->GetXaxis()->SetTitle("F_{90}");
  datasub_f90->GetXaxis()->CenterTitle();
  datasub_f90->GetXaxis()->SetTitleSize(0.045);
  datasub_f90->GetXaxis()->SetLabelSize(0.040);
  datasub_f90->GetXaxis()->SetTitleOffset(0.9);
  datasub_f90->GetYaxis()->SetLabelSize(0.045);
  maxaxis = std::max(datasub_f90->GetMaximum(), brnpluscevns_f90->GetMaximum());
  datasub_f90->GetYaxis()->SetRangeUser(-25.0, maxaxis*1.15);
  datasub_f90->Draw("p e0 x0");
  for(int i = 0; i < psderr.size(); ++i)
    psderr[i]->Draw("same");
  cevns_f90->Draw("hist same");
  brn_f90->Draw("hist same");
  brnpluscevns_f90->Draw("hist same");

  c2->Print("bkgsubf90ExtractedData.pdf");

  TCanvas *c3 = new TCanvas();
  datasub_time->SetStats(0);
  cevns_time->SetStats(0);
  brn_time->SetStats(0);
  brnpluscevns_time->SetStats(0);
  datasub_time->SetMarkerColor(kBlack);
  datasub_time->SetMarkerStyle(20);
  datasub_time->SetLineColor(kBlack);
  datasub_time->SetMarkerSize(1.5);
  cevns_time->SetLineWidth(2);
  cevns_time->SetLineColor(12);
  brn_time->SetLineWidth(2);
  brn_time->SetLineColor(11);
  brnpluscevns->SetLineWidth(2);
  brnpluscevns_time->SetLineColor(10);
  datasub_time->SetTitle("");
  datasub_time->GetXaxis()->SetTitle("t_{trig} (#mus)");
  datasub_time->GetXaxis()->CenterTitle();
  datasub_time->GetXaxis()->SetTitleSize(0.045);
  datasub_time->GetXaxis()->SetLabelSize(0.040);
  datasub_time->GetXaxis()->SetTitleOffset(0.9);
  datasub_time->GetYaxis()->SetLabelSize(0.045);
  datasub_time->GetYaxis()->SetTitle("SS-Background Subtracted Events");
  datasub_time->GetYaxis()->CenterTitle();
  datasub_time->GetYaxis()->SetTitleSize(0.045);
  datasub_time->GetYaxis()->SetTitleOffset(0.9);
  maxaxis = std::max(datasub_time->GetMaximum(), brnpluscevns_time->GetMaximum());
  datasub_time->GetYaxis()->SetRangeUser(-40.0, maxaxis*1.15);
  datasub_time->Draw("p e0 x0");
  for(int i = 0; i < timeerr.size(); ++i)
    timeerr[i]->Draw("same");
  cevns_time->Draw("hist same");
  brn_time->Draw("hist same");
  brnpluscevns_time->Draw("hist same");

  TLegend *leg2 = new TLegend(0.55,0.50,0.85,0.85);
  leg2->SetBorderSize(0);
  leg2->SetFillColor(0);
  leg2->AddEntry(datasub_time, "Data", "p");
  leg2->AddEntry(brnpluscevns_time, "Total", "L");
  leg2->AddEntry(cevns_time, "CEvNS", "L");
  leg2->AddEntry(brn_time, "BRN", "L");
  leg2->AddEntry(timeerr[0], "Syst. Error", "F");
  leg2->Draw("same");

  c3->Print("bkgsubtimeExtractedData.pdf");
}
