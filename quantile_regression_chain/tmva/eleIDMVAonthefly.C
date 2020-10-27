#include <iostream>
using namespace std;


struct eleIDInput{
  float ele_oldsigmaietaieta;
  float ele_oldsigmaiphiiphi;
  float ele_oldcircularity;
  float ele_oldr9;
  float ele_scletawidth;
  float ele_sclphiwidth;
  float ele_oldhe;
  float ele_kfhits;
  float ele_kfchi2;
  float ele_gsfchi2;
  float ele_fbrem;
  float ele_gsfhits;
  float ele_expected_inner_hits;
  float ele_conversionVertexFitProbability;
  float ele_ep;
  float ele_eelepout;
  float ele_IoEmIop;
  float ele_deltaetain;
  float ele_deltaphiin;
  float ele_deltaetaseed;
  float ele_pfPhotonIso;
  float ele_pfChargedHadIso;
  float ele_pfNeutralHadIso;
  float ele_psEoverEraw;
  float rho;
  float ele_pt;
  float scl_eta;
};



TMVA::Reader* bookReadersEB(const string &xmlfilenameEB, eleIDInput &inp, bool leg2016=false){
  // **** bdt 2015 EB ****
  //cout << "inside" << endl;

  string mvamethod = "BDT";
  
  TMVA::Reader* eleIdMva_EB_ = new TMVA::Reader( "!Color:Silent" );
  
  eleIdMva_EB_->AddVariable( "ele_oldsigmaietaieta", &inp.ele_oldsigmaietaieta);
  eleIdMva_EB_->AddVariable( "ele_oldsigmaiphiiphi", &inp.ele_oldsigmaiphiiphi);
  eleIdMva_EB_->AddVariable( "ele_oldcircularity", &inp.ele_oldcircularity);
  eleIdMva_EB_->AddVariable( "ele_oldr9", &inp.ele_oldr9);
  eleIdMva_EB_->AddVariable( "ele_scletawidth", &inp.ele_scletawidth);
  eleIdMva_EB_->AddVariable( "ele_sclphiwidth", &inp.ele_sclphiwidth);
  eleIdMva_EB_->AddVariable( "ele_oldhe", &inp.ele_oldhe);
  eleIdMva_EB_->AddVariable( "ele_kfhits", &inp.ele_kfhits);
  eleIdMva_EB_->AddVariable( "ele_kfchi2", &inp.ele_kfchi2);
  eleIdMva_EB_->AddVariable( "ele_gsfchi2", &inp.ele_gsfchi2);
  eleIdMva_EB_->AddVariable( "ele_fbrem", &inp.ele_fbrem);
  eleIdMva_EB_->AddVariable( "ele_gsfhits", &inp.ele_gsfhits);
  eleIdMva_EB_->AddVariable( "ele_expected_inner_hits", &inp.ele_expected_inner_hits);
  eleIdMva_EB_->AddVariable( "ele_conversionVertexFitProbability", &inp.ele_conversionVertexFitProbability);
  eleIdMva_EB_->AddVariable( "ele_ep", &inp.ele_ep);
  eleIdMva_EB_->AddVariable( "ele_eelepout", &inp.ele_eelepout);
  eleIdMva_EB_->AddVariable( "ele_IoEmIop", &inp.ele_IoEmIop);
  eleIdMva_EB_->AddVariable( "ele_deltaetain", &inp.ele_deltaetain);
  eleIdMva_EB_->AddVariable( "ele_deltaphiin", &inp.ele_deltaphiin);
  eleIdMva_EB_->AddVariable( "ele_deltaetaseed", &inp.ele_deltaetaseed);
  if (leg2016){
    eleIdMva_EB_->AddVariable( "ele_pt", &inp.ele_pt);
    eleIdMva_EB_->AddVariable( "scl_eta", &inp.scl_eta);
  }
  else{
    eleIdMva_EB_->AddVariable( "rho", &inp.rho);
    eleIdMva_EB_->AddVariable( "ele_pfPhotonIso", &inp.ele_pfPhotonIso);
    eleIdMva_EB_->AddVariable( "ele_pfChargedHadIso", &inp.ele_pfChargedHadIso);
    eleIdMva_EB_->AddVariable( "ele_pfNeutralHadIso", &inp.ele_pfNeutralHadIso);
  }

  eleIdMva_EB_->BookMVA( mvamethod.c_str(), xmlfilenameEB );
  return eleIdMva_EB_;
}



TMVA::Reader* bookReadersEE(const string &xmlfilenameEE, eleIDInput &inp, bool leg2016=false){
  //cout << "inside" << endl;
  // **** bdt 2015 EE ****

  string mvamethod = "BDT";

  TMVA::Reader* eleIdMva_EE_ = new TMVA::Reader( "!Color:Silent" );
    
  eleIdMva_EE_->AddVariable( "ele_oldsigmaietaieta", &inp.ele_oldsigmaietaieta);
  eleIdMva_EE_->AddVariable( "ele_oldsigmaiphiiphi", &inp.ele_oldsigmaiphiiphi);
  eleIdMva_EE_->AddVariable( "ele_oldcircularity", &inp.ele_oldcircularity);
  eleIdMva_EE_->AddVariable( "ele_oldr9", &inp.ele_oldr9);
  eleIdMva_EE_->AddVariable( "ele_scletawidth", &inp.ele_scletawidth);
  eleIdMva_EE_->AddVariable( "ele_sclphiwidth", &inp.ele_sclphiwidth);
  eleIdMva_EE_->AddVariable( "ele_oldhe", &inp.ele_oldhe);
  eleIdMva_EE_->AddVariable( "ele_kfhits", &inp.ele_kfhits);
  eleIdMva_EE_->AddVariable( "ele_kfchi2", &inp.ele_kfchi2);
  eleIdMva_EE_->AddVariable( "ele_gsfchi2", &inp.ele_gsfchi2);
  eleIdMva_EE_->AddVariable( "ele_fbrem", &inp.ele_fbrem);
  eleIdMva_EE_->AddVariable( "ele_gsfhits", &inp.ele_gsfhits);
  eleIdMva_EE_->AddVariable( "ele_expected_inner_hits", &inp.ele_expected_inner_hits);
  eleIdMva_EE_->AddVariable( "ele_conversionVertexFitProbability", &inp.ele_conversionVertexFitProbability);
  eleIdMva_EE_->AddVariable( "ele_ep", &inp.ele_ep);
  eleIdMva_EE_->AddVariable( "ele_eelepout", &inp.ele_eelepout);
  eleIdMva_EE_->AddVariable( "ele_IoEmIop", &inp.ele_IoEmIop);
  eleIdMva_EE_->AddVariable( "ele_deltaetain", &inp.ele_deltaetain);
  eleIdMva_EE_->AddVariable( "ele_deltaphiin", &inp.ele_deltaphiin);
  eleIdMva_EE_->AddVariable( "ele_deltaetaseed", &inp.ele_deltaetaseed);
  if (leg2016){
    eleIdMva_EE_->AddVariable( "ele_pt", &inp.ele_pt);
    eleIdMva_EE_->AddVariable( "scl_eta", &inp.scl_eta);
  } 
  else{
    eleIdMva_EE_->AddVariable( "rho", &inp.rho);};
  eleIdMva_EE_->AddVariable( "ele_psEoverEraw", &inp.ele_psEoverEraw);
  if (!leg2016){
    eleIdMva_EE_->AddVariable( "ele_pfPhotonIso", &inp.ele_pfPhotonIso);
    eleIdMva_EE_->AddVariable( "ele_pfChargedHadIso", &inp.ele_pfChargedHadIso);
    eleIdMva_EE_->AddVariable( "ele_pfNeutralHadIso", &inp.ele_pfNeutralHadIso);
  }

  
  eleIdMva_EE_->BookMVA( mvamethod.c_str(), xmlfilenameEE );
  return eleIdMva_EE_;
  
}
