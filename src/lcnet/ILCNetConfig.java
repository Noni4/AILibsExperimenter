package lcnet;

import java.util.List;

import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IExperimentSetConfig;

@Sources({ "file:./setup.properties", "file:./database.properties"})
public interface ILCNetConfig extends IExperimentSetConfig {
	public static final String SEEDS = "seeds";
	public static final String DATASETS = "datasets";
	public static final String PATH = "path";
	public static final String DATASETSIZE = "datasetSize";
	
	@Key(SEEDS)
	public List getSeeds();
	
	@Key(DATASETS)
	public List getDatasets();
	
	@Key(DATASETSIZE)
	public List getDatasetSize();
	
	@Key(PATH)
	public List getPath();
}
