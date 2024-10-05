
class DescriptionTemplate(object):
    def __init__(self):
        self.description = self.description_template()
    
    def description_template(self):
        desc = {}

        desc['protein_protein'] = "Protein {A} interacts with protein {B}, indicating that the two proteins directly or indirectly associate with each other to perform a biological function."
        desc['carrier'] = "{A} acts as a carrier for {B}, facilitating its transport or delivery to specific locations within the body or within a cell."
        desc['enzyme'] = "{A} functions as an enzyme that catalyzes a reaction involving {B}, converting it into a different molecule or modifying its structure."
        desc['target'] = "{A} serves as a target for {B}, meaning that {B} binds to or interacts with {A} to exert its biological effect."
        desc['transporter'] = "{A} is a transporter that facilitates the movement of {B} across cellular membranes or within different compartments of the body."
        desc['contraindication'] = "The interaction between {A} and {B} is contraindicated, meaning that the presence of one molecule may have adverse effects or reduce the efficacy of the other."
        desc['indication'] = "{A} is indicated for the treatment or management of a condition associated with {B}, suggesting that {A} has a therapeutic role related to {B}."
        desc['off-label use'] = "{A} is used off-label in relation to {B}, meaning it is utilized in a manner not specifically approved but based on clinical judgment."
        desc['synergistic interaction'] = "{A} and {B} interact synergistically, where their combined effect is greater than the sum of their individual effects."
        desc['associated with'] = "{A} is associated with {B}, indicating a relationship or correlation between the two, often in the context of disease or biological processes."
        desc['parent-child'] = "{A} is related to {B} in a parent-child relationship, where {A} gives rise to or influences the formation of {B}."
        desc['phenotype absent'] = "The interaction between {A} and {B} results in the absence of a specific phenotype, indicating that the normal trait is not expressed."
        desc['phenotype present'] = "The interaction between {A} and {B} results in the presence of a specific phenotype, indicating that a particular trait is expressed."
        desc['side effect'] = "The interaction between {A} and {B} can cause a side effect, where the presence of one molecule leads to unintended and possibly adverse effects."
        desc['interacts with'] = "{A} interacts with {B}, indicating a general interaction that may involve binding, modulation, or other forms of molecular communication."
        desc['linked to'] = "{A} is linked to {B}, suggesting a connection or association between the two molecules, often in a biological or pathological context."
        desc['expression present'] = "{A} is expressed in the presence of {B}, indicating that the existence or activity of {B} leads to or correlates with the expression of {A}."
        desc['expression absent'] = "{A} is not expressed in the presence of {B}, indicating that the existence or activity of {B} suppresses or does not correlate with the expression of {A}."

        return desc
        

    def get_description(self, triple):
        head_entity, rel, tail_entity = triple[:3]

        return self.description[rel].replace('{A}', head_entity).replace('{B}', tail_entity)
        